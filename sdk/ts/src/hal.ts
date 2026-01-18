/**
 * HAL – Harmonic Attention Layer
 *
 * Integrates harmonic scaling into transformer-style attention.
 * The coupling matrix Λ modulates attention weights based on
 * harmonic distances between query and key positions.
 *
 * Attention(Q, K, V) = softmax((Q·K^T / √d) ⊙ Λ) · V
 *
 * where Λᵢⱼ = R^(d_Q[i] · d_K[j])
 */

import type { HALConfig, HALOutput, Tensor3D, Tensor2D, Matrix2D } from './types';
import { CONSTANTS } from './harmonic-scaling';

// ═══════════════════════════════════════════════════════════════
// Matrix Operations
// ═══════════════════════════════════════════════════════════════

/**
 * Compute harmonic coupling matrix.
 *
 * Λᵢⱼ = R^(d_Q[i] · d_K[j] - d_max) if normalize
 * Λᵢⱼ = R^(d_Q[i] · d_K[j]) otherwise
 *
 * @param d_Q - Query distance indicators
 * @param d_K - Key distance indicators
 * @param R - Harmonic ratio
 * @param normalize - Subtract max exponent for stability
 * @returns Coupling matrix
 */
export function harmonicCouplingMatrix(
  d_Q: number[],
  d_K: number[],
  R: number = CONSTANTS.R_FIFTH,
  normalize: boolean = true
): Matrix2D {
  if (!(R > 0)) throw new RangeError('R must be > 0');

  const n = d_Q.length;
  const m = d_K.length;
  const M: Matrix2D = Array.from({ length: n }, () => Array(m).fill(0));

  let dmax = 0;
  if (normalize) {
    const maxQ = Math.max(...d_Q);
    const maxK = Math.max(...d_K);
    dmax = maxQ * maxK;
  }

  const lnR = Math.log(R);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      const expo = d_Q[i] * d_K[j] - (normalize ? dmax : 0);
      M[i][j] = Math.exp(expo * lnR);
    }
  }

  return M;
}

/**
 * Row-wise softmax.
 */
function softmaxRowWise(M: Matrix2D): Matrix2D {
  return M.map((row) => {
    const maxVal = Math.max(...row);
    const shifted = row.map((x) => x - maxVal);
    const exps = shifted.map(Math.exp);
    const sum = exps.reduce((a, b) => a + b, 0) || 1;
    return exps.map((x) => x / sum);
  });
}

/**
 * Matrix multiplication.
 */
function matMul(A: Matrix2D, B: Matrix2D): Matrix2D {
  const n = A.length;
  const k = A[0]?.length ?? 0;
  const m = B[0]?.length ?? 0;

  if (k !== B.length) {
    throw new RangeError('matMul shape mismatch');
  }

  const C: Matrix2D = Array.from({ length: n }, () => Array(m).fill(0));

  for (let i = 0; i < n; i++) {
    for (let t = 0; t < k; t++) {
      const a = A[i][t];
      const Bt = B[t];
      for (let j = 0; j < m; j++) {
        C[i][j] += a * Bt[j];
      }
    }
  }

  return C;
}

/**
 * Matrix transpose.
 */
function transpose(A: Matrix2D): Matrix2D {
  const n = A.length;
  const m = A[0]?.length ?? 0;
  const T: Matrix2D = Array.from({ length: m }, () => Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      T[j][i] = A[i][j];
    }
  }

  return T;
}

/**
 * Element-wise matrix multiplication (Hadamard product).
 */
function hadamard(A: Matrix2D, B: Matrix2D): Matrix2D {
  return A.map((row, i) => row.map((val, j) => val * B[i][j]));
}

// ═══════════════════════════════════════════════════════════════
// HAL Attention
// ═══════════════════════════════════════════════════════════════

/**
 * Compute HAL (Harmonic Attention Layer) output.
 *
 * This modulates standard scaled dot-product attention with
 * harmonic coupling based on distance indicators.
 *
 * @param Q - Query tensor [batch, seq_q, d_model]
 * @param K - Key tensor [batch, seq_k, d_model]
 * @param V - Value tensor [batch, seq_k, d_model]
 * @param d_Q - Query distance indicators [batch, seq_q]
 * @param d_K - Key distance indicators [batch, seq_k]
 * @param config - HAL configuration
 * @returns Attention output with metadata
 */
export function halAttention(
  Q: Tensor3D,
  K: Tensor3D,
  V: Tensor3D,
  d_Q: Tensor2D,
  d_K: Tensor2D,
  config: HALConfig
): HALOutput {
  const B = Q.length;
  const d_model = config.d_model;
  const R = config.R ?? CONSTANTS.R_FIFTH;
  const normalize = config.normalize ?? true;

  const outputs: Tensor3D = [];
  const allWeights: Tensor3D = [];
  let lastCoupling: Matrix2D = [];

  for (let b = 0; b < B; b++) {
    const Qb = Q[b];
    const Kb = K[b];
    const Vb = V[b];

    const n = Qb.length;
    const m = Kb.length;

    if (!n || !m) {
      throw new RangeError('Empty Q/K sequences');
    }

    if (
      (Qb[0]?.length ?? 0) !== d_model ||
      (Kb[0]?.length ?? 0) !== d_model ||
      (Vb[0]?.length ?? 0) !== d_model
    ) {
      throw new RangeError('d_model mismatch');
    }

    // Compute scaled dot-product: S = Q·K^T / √d
    const S = matMul(Qb, transpose(Kb)).map((row) =>
      row.map((x) => x / Math.sqrt(d_model))
    );

    // Compute harmonic coupling matrix
    const Lambda = harmonicCouplingMatrix(d_Q[b], d_K[b], R, normalize);
    lastCoupling = Lambda;

    // Apply coupling: S ⊙ Λ
    const modulated = hadamard(S, Lambda);

    // Softmax
    const W = softmaxRowWise(modulated);
    allWeights.push(W);

    // Output: W · V
    const Y = matMul(W, Vb);
    outputs.push(Y);
  }

  return {
    output: outputs,
    attentionWeights: allWeights,
    couplingMatrix: lastCoupling,
    timestamp: Date.now(),
  };
}

/**
 * Multi-head HAL attention.
 *
 * Splits Q, K, V into multiple heads, applies HAL attention
 * to each, then concatenates.
 *
 * @param Q - Query tensor
 * @param K - Key tensor
 * @param V - Value tensor
 * @param d_Q - Query distance indicators
 * @param d_K - Key distance indicators
 * @param config - HAL configuration with n_heads
 * @returns Combined multi-head output
 */
export function multiHeadHAL(
  Q: Tensor3D,
  K: Tensor3D,
  V: Tensor3D,
  d_Q: Tensor2D,
  d_K: Tensor2D,
  config: HALConfig
): HALOutput {
  const n_heads = config.n_heads;
  const d_model = config.d_model;
  const d_head = Math.floor(d_model / n_heads);

  if (d_head * n_heads !== d_model) {
    throw new RangeError('d_model must be divisible by n_heads');
  }

  const headOutputs: Tensor3D[] = [];
  let totalWeights: Tensor3D = [];
  let lastCoupling: Matrix2D = [];

  // Process each head
  for (let h = 0; h < n_heads; h++) {
    const startIdx = h * d_head;
    const endIdx = startIdx + d_head;

    // Slice Q, K, V for this head
    const Q_h = Q.map((batch) =>
      batch.map((seq) => seq.slice(startIdx, endIdx))
    );
    const K_h = K.map((batch) =>
      batch.map((seq) => seq.slice(startIdx, endIdx))
    );
    const V_h = V.map((batch) =>
      batch.map((seq) => seq.slice(startIdx, endIdx))
    );

    const headConfig: HALConfig = { ...config, d_model: d_head };
    const headResult = halAttention(Q_h, K_h, V_h, d_Q, d_K, headConfig);

    headOutputs.push(headResult.output);
    if (h === 0) {
      totalWeights = headResult.attentionWeights;
      lastCoupling = headResult.couplingMatrix;
    }
  }

  // Concatenate head outputs
  const concatenated: Tensor3D = Q.map((_, batchIdx) => {
    const batchSeqLen = headOutputs[0][batchIdx].length;
    return Array.from({ length: batchSeqLen }, (_, seqIdx) => {
      const concatenatedRow: number[] = [];
      for (let h = 0; h < n_heads; h++) {
        concatenatedRow.push(...headOutputs[h][batchIdx][seqIdx]);
      }
      return concatenatedRow;
    });
  });

  return {
    output: concatenated,
    attentionWeights: totalWeights,
    couplingMatrix: lastCoupling,
    timestamp: Date.now(),
  };
}

// ═══════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════

/**
 * Generate distance indicators from position embeddings.
 *
 * Uses L2 norm from origin as distance indicator.
 *
 * @param positions - Position embeddings [batch, seq, d]
 * @returns Distance indicators [batch, seq]
 */
export function positionToDistance(positions: Tensor3D): Tensor2D {
  return positions.map((batch) =>
    batch.map((pos) => Math.sqrt(pos.reduce((sum, x) => sum + x * x, 0)))
  );
}

/**
 * Create sinusoidal position embeddings.
 *
 * @param seqLen - Sequence length
 * @param d_model - Model dimension
 * @param base - Base for frequency scaling
 * @returns Position embeddings [seqLen, d_model]
 */
export function sinusoidalPositions(
  seqLen: number,
  d_model: number,
  base: number = 10000
): Matrix2D {
  const positions: Matrix2D = [];

  for (let pos = 0; pos < seqLen; pos++) {
    const row: number[] = [];
    for (let i = 0; i < d_model; i++) {
      const angle = pos / Math.pow(base, (2 * Math.floor(i / 2)) / d_model);
      row.push(i % 2 === 0 ? Math.sin(angle) : Math.cos(angle));
    }
    positions.push(row);
  }

  return positions;
}

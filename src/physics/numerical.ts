/**
 * Numerical Methods Module
 * Production-grade numerical algorithms for scientific computing
 */

export type VectorFunction = (t: number, y: number[]) => number[];
export type ScalarFunction = (x: number) => number;
export type MultiVarFunction = (x: number[]) => number;

export interface ODEResult {
  t: number[];
  y: number[][];
  steps: number;
  evaluations: number;
}

export interface RootResult {
  root: number;
  iterations: number;
  converged: boolean;
  error: number;
}

export interface IntegrationResult {
  value: number;
  error: number;
  evaluations: number;
}

export interface OptimizationResult {
  minimum: number[];
  value: number;
  iterations: number;
  converged: boolean;
  gradient?: number[];
}

export interface InterpolationResult {
  evaluate: (x: number) => number;
  derivative: (x: number) => number;
  coefficients: number[];
}

/**
 * Numerical Methods for Scientific Computing
 */
export class NumericalMethods {

  // ============================================
  // ODE SOLVERS
  // ============================================

  /**
   * 4th-order Runge-Kutta (RK4) integrator
   * The workhorse of ODE solving - accurate and stable
   */
  static rk4(
    f: VectorFunction,
    y0: number[],
    tSpan: [number, number],
    dt: number
  ): ODEResult {
    const [t0, tf] = tSpan;
    const steps = Math.ceil((tf - t0) / dt);
    const t: number[] = [t0];
    const y: number[][] = [y0.slice()];
    let evaluations = 0;

    let currentY = y0.slice();
    let currentT = t0;

    for (let i = 0; i < steps; i++) {
      const h = Math.min(dt, tf - currentT);

      // RK4 stages
      const k1 = f(currentT, currentY);
      evaluations++;

      const y2 = currentY.map((yi, j) => yi + 0.5 * h * k1[j]);
      const k2 = f(currentT + 0.5 * h, y2);
      evaluations++;

      const y3 = currentY.map((yi, j) => yi + 0.5 * h * k2[j]);
      const k3 = f(currentT + 0.5 * h, y3);
      evaluations++;

      const y4 = currentY.map((yi, j) => yi + h * k3[j]);
      const k4 = f(currentT + h, y4);
      evaluations++;

      // Update
      currentY = currentY.map((yi, j) =>
        yi + (h / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])
      );
      currentT += h;

      t.push(currentT);
      y.push(currentY.slice());
    }

    return { t, y, steps, evaluations };
  }

  /**
   * Adaptive RK45 (Dormand-Prince) with error control
   * Adjusts step size for accuracy
   */
  static rk45Adaptive(
    f: VectorFunction,
    y0: number[],
    tSpan: [number, number],
    tolerance: number = 1e-6,
    maxStep: number = 0.1,
    minStep: number = 1e-10
  ): ODEResult {
    const [t0, tf] = tSpan;
    const t: number[] = [t0];
    const y: number[][] = [y0.slice()];
    let evaluations = 0;

    // Dormand-Prince coefficients
    const c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1];
    const a = [
      [],
      [1/5],
      [3/40, 9/40],
      [44/45, -56/15, 32/9],
      [19372/6561, -25360/2187, 64448/6561, -212/729],
      [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
      [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    ];
    const b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0];
    const b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40];

    let currentY = y0.slice();
    let currentT = t0;
    let h = Math.min(maxStep, (tf - t0) / 10);
    let steps = 0;

    while (currentT < tf) {
      h = Math.min(h, tf - currentT);
      if (h < minStep) h = minStep;

      // Compute stages
      const k: number[][] = [];
      k[0] = f(currentT, currentY);
      evaluations++;

      for (let i = 1; i <= 6; i++) {
        const yi = currentY.map((yj, idx) => {
          let sum = yj;
          for (let j = 0; j < i; j++) {
            sum += h * a[i][j] * k[j][idx];
          }
          return sum;
        });
        k[i] = f(currentT + c[i] * h, yi);
        evaluations++;
      }

      // 5th order solution
      const y5 = currentY.map((yj, idx) => {
        let sum = yj;
        for (let i = 0; i < 7; i++) {
          sum += h * b5[i] * k[i][idx];
        }
        return sum;
      });

      // 4th order solution (for error estimate)
      const y4 = currentY.map((yj, idx) => {
        let sum = yj;
        for (let i = 0; i < 7; i++) {
          sum += h * b4[i] * k[i][idx];
        }
        return sum;
      });

      // Error estimate
      const error = Math.max(...y5.map((v, i) => Math.abs(v - y4[i])));

      if (error <= tolerance || h <= minStep) {
        // Accept step
        currentT += h;
        currentY = y5;
        t.push(currentT);
        y.push(currentY.slice());
        steps++;
      }

      // Adjust step size
      if (error > 0) {
        const factor = 0.9 * Math.pow(tolerance / error, 0.2);
        h = Math.max(minStep, Math.min(maxStep, h * Math.min(4, Math.max(0.1, factor))));
      }
    }

    return { t, y, steps, evaluations };
  }

  /**
   * Velocity Verlet integrator (symplectic - conserves energy)
   * Perfect for Hamiltonian systems (orbital mechanics, molecular dynamics)
   */
  static velocityVerlet(
    acceleration: (pos: number[], vel: number[]) => number[],
    pos0: number[],
    vel0: number[],
    tSpan: [number, number],
    dt: number
  ): { t: number[]; position: number[][]; velocity: number[][] } {
    const [t0, tf] = tSpan;
    const steps = Math.ceil((tf - t0) / dt);

    const t: number[] = [t0];
    const position: number[][] = [pos0.slice()];
    const velocity: number[][] = [vel0.slice()];

    let pos = pos0.slice();
    let vel = vel0.slice();
    let acc = acceleration(pos, vel);

    for (let i = 0; i < steps; i++) {
      const h = Math.min(dt, tf - t[t.length - 1]);

      // Update position: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
      const newPos = pos.map((p, j) => p + vel[j] * h + 0.5 * acc[j] * h * h);

      // Calculate new acceleration
      const newAcc = acceleration(newPos, vel);

      // Update velocity: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
      const newVel = vel.map((v, j) => v + 0.5 * (acc[j] + newAcc[j]) * h);

      pos = newPos;
      vel = newVel;
      acc = newAcc;

      t.push(t[t.length - 1] + h);
      position.push(pos.slice());
      velocity.push(vel.slice());
    }

    return { t, position, velocity };
  }

  // ============================================
  // ROOT FINDING
  // ============================================

  /**
   * Newton-Raphson method for finding roots
   */
  static newtonRaphson(
    f: ScalarFunction,
    df: ScalarFunction,
    x0: number,
    tolerance: number = 1e-10,
    maxIterations: number = 100
  ): RootResult {
    let x = x0;
    let iterations = 0;

    while (iterations < maxIterations) {
      const fx = f(x);
      const dfx = df(x);

      if (Math.abs(dfx) < 1e-15) {
        return { root: x, iterations, converged: false, error: Math.abs(fx) };
      }

      const xNew = x - fx / dfx;
      const error = Math.abs(xNew - x);

      if (error < tolerance) {
        return { root: xNew, iterations: iterations + 1, converged: true, error };
      }

      x = xNew;
      iterations++;
    }

    return { root: x, iterations, converged: false, error: Math.abs(f(x)) };
  }

  /**
   * Brent's method - robust bracketing root finder
   * Combines bisection, secant, and inverse quadratic interpolation
   */
  static brent(
    f: ScalarFunction,
    a: number,
    b: number,
    tolerance: number = 1e-10,
    maxIterations: number = 100
  ): RootResult {
    let fa = f(a);
    let fb = f(b);

    if (fa * fb > 0) {
      throw new Error('Function must have opposite signs at interval endpoints');
    }

    if (Math.abs(fa) < Math.abs(fb)) {
      [a, b] = [b, a];
      [fa, fb] = [fb, fa];
    }

    let c = a;
    let fc = fa;
    let d = b - a;
    let e = d;
    let iterations = 0;

    while (iterations < maxIterations) {
      if (Math.abs(fb) < tolerance) {
        return { root: b, iterations, converged: true, error: Math.abs(fb) };
      }

      if (fa !== fc && fb !== fc) {
        // Inverse quadratic interpolation
        const s = a * fb * fc / ((fa - fb) * (fa - fc)) +
                  b * fa * fc / ((fb - fa) * (fb - fc)) +
                  c * fa * fb / ((fc - fa) * (fc - fb));

        // Check if interpolation is acceptable
        const cond1 = (s - b) * (s - (3 * a + b) / 4) > 0;
        const cond2 = Math.abs(s - b) >= Math.abs(e) / 2;
        const cond3 = Math.abs(e) < tolerance;

        if (cond1 || cond2 || cond3) {
          // Bisection
          d = (a - b) / 2;
          e = d;
        } else {
          e = d;
          d = s - b;
        }
      } else {
        // Secant method
        d = (a - b) * fb / (fb - fa);
        e = d;
      }

      c = b;
      fc = fb;

      if (Math.abs(d) > tolerance) {
        b = b + d;
      } else {
        b = b + (a > b ? tolerance : -tolerance);
      }

      fb = f(b);

      if (fb * fc > 0) {
        c = a;
        fc = fa;
        d = b - a;
        e = d;
      }

      if (Math.abs(fc) < Math.abs(fb)) {
        a = b;
        b = c;
        c = a;
        fa = fb;
        fb = fc;
        fc = fa;
      } else {
        a = c;
        fa = fc;
      }

      iterations++;
    }

    return { root: b, iterations, converged: false, error: Math.abs(fb) };
  }

  /**
   * Bisection method - guaranteed convergence
   */
  static bisection(
    f: ScalarFunction,
    a: number,
    b: number,
    tolerance: number = 1e-10,
    maxIterations: number = 100
  ): RootResult {
    let fa = f(a);
    let fb = f(b);

    if (fa * fb > 0) {
      throw new Error('Function must have opposite signs at interval endpoints');
    }

    let iterations = 0;

    while (iterations < maxIterations) {
      const c = (a + b) / 2;
      const fc = f(c);

      if (Math.abs(fc) < tolerance || (b - a) / 2 < tolerance) {
        return { root: c, iterations, converged: true, error: Math.abs(fc) };
      }

      if (fa * fc < 0) {
        b = c;
        fb = fc;
      } else {
        a = c;
        fa = fc;
      }

      iterations++;
    }

    return { root: (a + b) / 2, iterations, converged: false, error: Math.abs(f((a + b) / 2)) };
  }

  // ============================================
  // NUMERICAL INTEGRATION
  // ============================================

  /**
   * Simpson's rule integration
   */
  static simpsons(
    f: ScalarFunction,
    a: number,
    b: number,
    n: number = 100
  ): IntegrationResult {
    if (n % 2 !== 0) n++;  // Must be even

    const h = (b - a) / n;
    let sum = f(a) + f(b);
    let evaluations = 2;

    for (let i = 1; i < n; i++) {
      const x = a + i * h;
      sum += (i % 2 === 0 ? 2 : 4) * f(x);
      evaluations++;
    }

    const value = (h / 3) * sum;

    // Error estimate using Richardson extrapolation
    const halfN = n / 2;
    const h2 = (b - a) / halfN;
    let sum2 = f(a) + f(b);
    for (let i = 1; i < halfN; i++) {
      sum2 += (i % 2 === 0 ? 2 : 4) * f(a + i * h2);
    }
    const value2 = (h2 / 3) * sum2;
    const error = Math.abs(value - value2) / 15;

    return { value, error, evaluations };
  }

  /**
   * Gaussian quadrature (5-point Gauss-Legendre)
   * Extremely accurate for smooth functions
   */
  static gaussianQuadrature(
    f: ScalarFunction,
    a: number,
    b: number,
    intervals: number = 10
  ): IntegrationResult {
    // 5-point Gauss-Legendre nodes and weights
    const nodes = [
      -0.9061798459386640,
      -0.5384693101056831,
      0.0,
      0.5384693101056831,
      0.9061798459386640
    ];
    const weights = [
      0.2369268850561891,
      0.4786286704993665,
      0.5688888888888889,
      0.4786286704993665,
      0.2369268850561891
    ];

    let total = 0;
    let evaluations = 0;
    const h = (b - a) / intervals;

    for (let i = 0; i < intervals; i++) {
      const x0 = a + i * h;
      const x1 = x0 + h;
      const mid = (x0 + x1) / 2;
      const halfWidth = (x1 - x0) / 2;

      for (let j = 0; j < 5; j++) {
        const x = mid + halfWidth * nodes[j];
        total += weights[j] * f(x) * halfWidth;
        evaluations++;
      }
    }

    // Simple error estimate
    const error = Math.abs(total) * 1e-10;

    return { value: total, error, evaluations };
  }

  /**
   * Adaptive Simpson's integration with error control
   */
  static adaptiveSimpson(
    f: ScalarFunction,
    a: number,
    b: number,
    tolerance: number = 1e-8,
    maxDepth: number = 50
  ): IntegrationResult {
    let evaluations = 0;

    const simpson = (a: number, b: number): number => {
      const h = (b - a) / 2;
      const fa = f(a);
      const fm = f((a + b) / 2);
      const fb = f(b);
      evaluations += 3;
      return (h / 3) * (fa + 4 * fm + fb);
    };

    const adaptiveHelper = (a: number, b: number, whole: number, tol: number, depth: number): number => {
      const c = (a + b) / 2;
      const left = simpson(a, c);
      const right = simpson(c, b);
      const delta = left + right - whole;

      if (depth >= maxDepth || Math.abs(delta) <= 15 * tol) {
        return left + right + delta / 15;
      }

      return adaptiveHelper(a, c, left, tol / 2, depth + 1) +
             adaptiveHelper(c, b, right, tol / 2, depth + 1);
    };

    const whole = simpson(a, b);
    const value = adaptiveHelper(a, b, whole, tolerance, 0);

    return { value, error: tolerance, evaluations };
  }

  // ============================================
  // INTERPOLATION
  // ============================================

  /**
   * Lagrange polynomial interpolation
   */
  static lagrangeInterpolation(
    xData: number[],
    yData: number[]
  ): InterpolationResult {
    const n = xData.length;

    const evaluate = (x: number): number => {
      let result = 0;

      for (let i = 0; i < n; i++) {
        let term = yData[i];
        for (let j = 0; j < n; j++) {
          if (i !== j) {
            term *= (x - xData[j]) / (xData[i] - xData[j]);
          }
        }
        result += term;
      }

      return result;
    };

    const derivative = (x: number): number => {
      const h = 1e-8;
      return (evaluate(x + h) - evaluate(x - h)) / (2 * h);
    };

    return { evaluate, derivative, coefficients: yData };
  }

  /**
   * Cubic spline interpolation
   */
  static cubicSpline(
    xData: number[],
    yData: number[]
  ): InterpolationResult {
    const n = xData.length;
    const h: number[] = [];
    const alpha: number[] = [];
    const l: number[] = [1];
    const mu: number[] = [0];
    const z: number[] = [0];
    const c: number[] = new Array(n).fill(0);
    const b: number[] = new Array(n - 1);
    const d: number[] = new Array(n - 1);

    // Calculate h and alpha
    for (let i = 0; i < n - 1; i++) {
      h[i] = xData[i + 1] - xData[i];
    }

    for (let i = 1; i < n - 1; i++) {
      alpha[i] = (3 / h[i]) * (yData[i + 1] - yData[i]) -
                 (3 / h[i - 1]) * (yData[i] - yData[i - 1]);
    }

    // Solve tridiagonal system
    for (let i = 1; i < n - 1; i++) {
      l[i] = 2 * (xData[i + 1] - xData[i - 1]) - h[i - 1] * mu[i - 1];
      mu[i] = h[i] / l[i];
      z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    l[n - 1] = 1;
    z[n - 1] = 0;

    // Back substitution
    for (let j = n - 2; j >= 0; j--) {
      c[j] = z[j] - mu[j] * c[j + 1];
      b[j] = (yData[j + 1] - yData[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3;
      d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
    }

    const evaluate = (x: number): number => {
      // Find interval
      let i = 0;
      for (let j = 0; j < n - 1; j++) {
        if (x >= xData[j] && x <= xData[j + 1]) {
          i = j;
          break;
        }
      }

      const dx = x - xData[i];
      return yData[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx;
    };

    const derivative = (x: number): number => {
      let i = 0;
      for (let j = 0; j < n - 1; j++) {
        if (x >= xData[j] && x <= xData[j + 1]) {
          i = j;
          break;
        }
      }

      const dx = x - xData[i];
      return b[i] + 2 * c[i] * dx + 3 * d[i] * dx * dx;
    };

    return { evaluate, derivative, coefficients: [...b, ...c, ...d] };
  }

  // ============================================
  // LINEAR ALGEBRA
  // ============================================

  /**
   * Gaussian elimination with partial pivoting
   */
  static solveLinearSystem(A: number[][], b: number[]): number[] {
    const n = A.length;
    const aug: number[][] = A.map((row, i) => [...row, b[i]]);

    // Forward elimination with partial pivoting
    for (let k = 0; k < n; k++) {
      // Find pivot
      let maxIdx = k;
      for (let i = k + 1; i < n; i++) {
        if (Math.abs(aug[i][k]) > Math.abs(aug[maxIdx][k])) {
          maxIdx = i;
        }
      }
      [aug[k], aug[maxIdx]] = [aug[maxIdx], aug[k]];

      if (Math.abs(aug[k][k]) < 1e-15) {
        throw new Error('Matrix is singular or nearly singular');
      }

      // Eliminate
      for (let i = k + 1; i < n; i++) {
        const factor = aug[i][k] / aug[k][k];
        for (let j = k; j <= n; j++) {
          aug[i][j] -= factor * aug[k][j];
        }
      }
    }

    // Back substitution
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      let sum = aug[i][n];
      for (let j = i + 1; j < n; j++) {
        sum -= aug[i][j] * x[j];
      }
      x[i] = sum / aug[i][i];
    }

    return x;
  }

  /**
   * Matrix determinant using LU decomposition
   */
  static determinant(A: number[][]): number {
    const n = A.length;
    const L: number[][] = Array(n).fill(0).map(() => Array(n).fill(0));
    const U: number[][] = A.map(row => [...row]);
    let det = 1;

    for (let k = 0; k < n; k++) {
      // Find pivot
      let maxIdx = k;
      for (let i = k + 1; i < n; i++) {
        if (Math.abs(U[i][k]) > Math.abs(U[maxIdx][k])) {
          maxIdx = i;
        }
      }

      if (maxIdx !== k) {
        [U[k], U[maxIdx]] = [U[maxIdx], U[k]];
        det *= -1;
      }

      if (Math.abs(U[k][k]) < 1e-15) {
        return 0;
      }

      det *= U[k][k];

      for (let i = k + 1; i < n; i++) {
        const factor = U[i][k] / U[k][k];
        for (let j = k; j < n; j++) {
          U[i][j] -= factor * U[k][j];
        }
      }
    }

    return det;
  }

  /**
   * Eigenvalues using QR iteration (for small matrices)
   */
  static eigenvalues(A: number[][], maxIterations: number = 100): number[] {
    const n = A.length;
    let H = A.map(row => [...row]);

    for (let iter = 0; iter < maxIterations; iter++) {
      // QR decomposition using Gram-Schmidt
      const Q: number[][] = Array(n).fill(0).map(() => Array(n).fill(0));
      const R: number[][] = Array(n).fill(0).map(() => Array(n).fill(0));

      for (let j = 0; j < n; j++) {
        const v = H.map(row => row[j]);

        for (let i = 0; i < j; i++) {
          R[i][j] = Q.reduce((sum, row, k) => sum + row[i] * v[k], 0);
          for (let k = 0; k < n; k++) {
            v[k] -= R[i][j] * Q[k][i];
          }
        }

        R[j][j] = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));

        if (R[j][j] > 1e-10) {
          for (let k = 0; k < n; k++) {
            Q[k][j] = v[k] / R[j][j];
          }
        }
      }

      // H = RQ
      const newH: number[][] = Array(n).fill(0).map(() => Array(n).fill(0));
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          for (let k = 0; k < n; k++) {
            newH[i][j] += R[i][k] * Q[k][j];
          }
        }
      }
      H = newH;
    }

    // Eigenvalues are on the diagonal
    return H.map((row, i) => row[i]);
  }

  // ============================================
  // FINITE DIFFERENCES
  // ============================================

  /**
   * Numerical derivative using central difference
   */
  static derivative(f: ScalarFunction, x: number, h: number = 1e-8): number {
    return (f(x + h) - f(x - h)) / (2 * h);
  }

  /**
   * Second derivative using central difference
   */
  static secondDerivative(f: ScalarFunction, x: number, h: number = 1e-5): number {
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h);
  }

  /**
   * Gradient of multivariate function
   */
  static gradient(f: MultiVarFunction, x: number[], h: number = 1e-8): number[] {
    const n = x.length;
    const grad: number[] = [];

    for (let i = 0; i < n; i++) {
      const xPlus = [...x];
      const xMinus = [...x];
      xPlus[i] += h;
      xMinus[i] -= h;
      grad.push((f(xPlus) - f(xMinus)) / (2 * h));
    }

    return grad;
  }

  /**
   * Hessian matrix of multivariate function
   */
  static hessian(f: MultiVarFunction, x: number[], h: number = 1e-5): number[][] {
    const n = x.length;
    const H: number[][] = Array(n).fill(0).map(() => Array(n).fill(0));

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const x1 = [...x]; x1[i] += h; x1[j] += h;
        const x2 = [...x]; x2[i] += h; x2[j] -= h;
        const x3 = [...x]; x3[i] -= h; x3[j] += h;
        const x4 = [...x]; x4[i] -= h; x4[j] -= h;

        H[i][j] = (f(x1) - f(x2) - f(x3) + f(x4)) / (4 * h * h);
      }
    }

    return H;
  }
}

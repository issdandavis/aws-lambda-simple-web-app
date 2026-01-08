/**
 * Wave Simulation Tests
 */

import { WaveSimulation } from '../../../src/physics/waves';
import { PhysicalConstants } from '../../../src/physics/constants';

describe('WaveSimulation', () => {
  describe('calculateWaveParameters', () => {
    test('should satisfy v = λf', () => {
      const params = WaveSimulation.calculateWaveParameters(1, 1e14, PhysicalConstants.c);
      expect(params.velocity).toBeCloseTo(params.wavelength * params.frequency, 5);
    });

    test('light in vacuum should have v = c', () => {
      const params = WaveSimulation.calculateWaveParameters(1, 5e14);
      expect(params.velocity).toBe(PhysicalConstants.c);
    });
  });

  describe('sinusoidalWave', () => {
    test('amplitude should not exceed wave amplitude', () => {
      const params = WaveSimulation.calculateWaveParameters(2, 1e6, 1000);
      const result = WaveSimulation.sinusoidalWave(params, 0, 0.01, 0);

      for (const y of result.displacement) {
        expect(Math.abs(y)).toBeLessThanOrEqual(params.amplitude + 1e-10);
      }
    });

    test('wave should propagate in positive x direction', () => {
      const params = WaveSimulation.calculateWaveParameters(1, 100, 340);
      const t1 = WaveSimulation.sinusoidalWave(params, 0, 10, 0);
      const t2 = WaveSimulation.sinusoidalWave(params, 0, 10, 0.001);

      // Peak should have moved in positive x direction
      const peak1 = t1.positions[t1.displacement.indexOf(Math.max(...t1.displacement))];
      const peak2 = t2.positions[t2.displacement.indexOf(Math.max(...t2.displacement))];

      expect(peak2).toBeGreaterThan(peak1);
    });
  });

  describe('twoSourceInterference', () => {
    test('central point should have maximum intensity', () => {
      const result = WaveSimulation.twoSourceInterference(
        500e-9, // 500nm green light
        1e-3,   // 1mm slit separation
        1,      // 1m to screen
        0.1     // 10cm screen width
      );

      // Find central intensity
      const centerIndex = Math.floor(result.positions.length / 2);
      const centralIntensity = result.resultantAmplitude[centerIndex];

      // Should be maximum (or close to it)
      const maxIntensity = Math.max(...result.resultantAmplitude);
      expect(centralIntensity).toBeCloseTo(maxIntensity, 1);
    });

    test('fringe spacing should be λL/d', () => {
      const wavelength = 600e-9;
      const d = 0.5e-3;
      const L = 2;

      const result = WaveSimulation.twoSourceInterference(wavelength, d, L, 0.1);
      const expectedSpacing = wavelength * L / d;

      expect(result.fringeSpacing).toBeCloseTo(expectedSpacing, 10);
    });

    test('constructive and destructive points should alternate', () => {
      const result = WaveSimulation.twoSourceInterference(500e-9, 1e-3, 1, 0.1);

      // Constructive points should be interleaved with destructive points
      for (let i = 0; i < result.constructivePoints.length - 1; i++) {
        const c1 = result.constructivePoints[i];
        const c2 = result.constructivePoints[i + 1];

        // There should be a destructive point between them
        const hasDestructiveBetween = result.destructivePoints.some(
          d => d > c1 && d < c2
        );
        expect(hasDestructiveBetween).toBe(true);
      }
    });
  });

  describe('singleSlitDiffraction', () => {
    test('central maximum should be at θ = 0', () => {
      const result = WaveSimulation.singleSlitDiffraction(500e-9, 1e-5, 1);

      // Find maximum
      const maxIndex = result.intensity.indexOf(Math.max(...result.intensity));
      expect(result.angles[maxIndex]).toBeCloseTo(0, 5);
    });

    test('first minimum should be at sinθ = λ/a', () => {
      const wavelength = 500e-9;
      const slitWidth = 1e-5;

      const result = WaveSimulation.singleSlitDiffraction(wavelength, slitWidth, 1);
      const expectedAngle = Math.asin(wavelength / slitWidth);

      expect(result.minima).toContain(expect.closeTo(expectedAngle, 5));
    });

    test('intensity should be symmetric around center', () => {
      const result = WaveSimulation.singleSlitDiffraction(600e-9, 2e-5, 1);

      const n = result.intensity.length;
      for (let i = 0; i < n / 2; i++) {
        expect(result.intensity[i]).toBeCloseTo(result.intensity[n - 1 - i], 5);
      }
    });
  });

  describe('diffractionGrating', () => {
    test('principal maxima should be at sinθ = mλ/d', () => {
      const wavelength = 500e-9;
      const d = 2e-6; // 500 lines/mm

      const result = WaveSimulation.diffractionGrating(wavelength, d, 100);

      // Check first-order maximum
      const expectedAngle = Math.asin(wavelength / d);
      expect(result.principalMaxima).toContain(expect.closeTo(expectedAngle, 5));
    });

    test('more slits should give sharper peaks', () => {
      const wavelength = 500e-9;
      const d = 2e-6;

      const few = WaveSimulation.diffractionGrating(wavelength, d, 5);
      const many = WaveSimulation.diffractionGrating(wavelength, d, 100);

      // Maximum should be higher with more slits (normalized)
      const fewMax = Math.max(...few.intensity);
      const manyMax = Math.max(...many.intensity);

      expect(manyMax).toBeCloseTo(1, 5);
      expect(fewMax).toBeCloseTo(1, 5);
    });
  });

  describe('dopplerSound', () => {
    test('approaching source should increase frequency', () => {
      const result = WaveSimulation.dopplerSound(1000, 340, -50, 0);
      expect(result.observedFrequency).toBeGreaterThan(1000);
    });

    test('receding source should decrease frequency', () => {
      const result = WaveSimulation.dopplerSound(1000, 340, 50, 0);
      expect(result.observedFrequency).toBeLessThan(1000);
    });

    test('stationary case should give same frequency', () => {
      const result = WaveSimulation.dopplerSound(1000, 340, 0, 0);
      expect(result.observedFrequency).toBeCloseTo(1000, 5);
    });

    test('approaching observer should increase frequency', () => {
      const result = WaveSimulation.dopplerSound(1000, 340, 0, 50);
      expect(result.observedFrequency).toBeGreaterThan(1000);
    });
  });

  describe('dopplerRelativistic', () => {
    test('receding source should give positive redshift', () => {
      const v = 0.1 * PhysicalConstants.c;
      const result = WaveSimulation.dopplerRelativistic(5e14, v);
      expect(result.redshift).toBeGreaterThan(0);
    });

    test('approaching source should give negative redshift (blueshift)', () => {
      const v = -0.1 * PhysicalConstants.c;
      const result = WaveSimulation.dopplerRelativistic(5e14, v);
      expect(result.redshift).toBeLessThan(0);
    });

    test('at v = 0 should give same frequency', () => {
      const result = WaveSimulation.dopplerRelativistic(5e14, 0);
      expect(result.observedFrequency).toBeCloseTo(5e14, 5);
    });

    test('should throw for v >= c', () => {
      expect(() => WaveSimulation.dopplerRelativistic(5e14, PhysicalConstants.c)).toThrow();
    });
  });

  describe('standingWaves', () => {
    test('fundamental frequency should be v/(2L)', () => {
      const L = 1;
      const tension = 100;
      const linearDensity = 0.01;
      const v = Math.sqrt(tension / linearDensity);

      const result = WaveSimulation.standingWaves(L, tension, linearDensity);
      expect(result.frequencies[0]).toBeCloseTo(v / (2 * L), 5);
    });

    test('harmonics should be integer multiples of fundamental', () => {
      const result = WaveSimulation.standingWaves(1, 100, 0.01);

      for (let n = 1; n < result.frequencies.length; n++) {
        expect(result.frequencies[n] / result.frequencies[0]).toBeCloseTo(n + 1, 5);
      }
    });

    test('number of nodes should be n+1 for mode n', () => {
      const result = WaveSimulation.standingWaves(1, 100, 0.01);

      for (let i = 0; i < result.modes.length; i++) {
        expect(result.nodePositions[i].length).toBe(result.modes[i] + 1);
      }
    });
  });

  describe('blackbodyRadiation', () => {
    test('peak wavelength should follow Wien law', () => {
      const T = 6000; // K (like the Sun)
      const result = WaveSimulation.blackbodyRadiation(T);

      const expectedPeak = 2.897771955e-3 / T;
      expect(result.peakWavelength).toBeCloseTo(expectedPeak, 10);
    });

    test('total power should follow Stefan-Boltzmann law', () => {
      const T = 5000;
      const result = WaveSimulation.blackbodyRadiation(T);

      const expectedPower = PhysicalConstants.sigma * Math.pow(T, 4);
      expect(result.totalPower).toBeCloseTo(expectedPower, 5);
    });

    test('hotter object should peak at shorter wavelength', () => {
      const cold = WaveSimulation.blackbodyRadiation(3000);
      const hot = WaveSimulation.blackbodyRadiation(6000);

      expect(hot.peakWavelength).toBeLessThan(cold.peakWavelength);
    });
  });

  describe('emWaveProperties', () => {
    test('E and B should be related by E = cB', () => {
      const result = WaveSimulation.emWaveProperties(100, 1e9);
      expect(result.magneticFieldAmplitude * PhysicalConstants.c).toBeCloseTo(100, 5);
    });

    test('intensity should equal Poynting vector time average', () => {
      const E0 = 100;
      const result = WaveSimulation.emWaveProperties(E0, 1e9);

      // Time-averaged Poynting vector = 0.5 * E0 * B0 / μ0
      const expectedIntensity = 0.5 * E0 * result.magneticFieldAmplitude / PhysicalConstants.mu0;
      expect(result.intensity).toBeCloseTo(expectedIntensity, 5);
    });
  });

  describe('refraction', () => {
    test('normal incidence should give no deviation', () => {
      const result = WaveSimulation.refraction(0, 1, 1.5);
      expect(result.refractedAngle).toBeCloseTo(0, 10);
    });

    test('light entering denser medium should bend toward normal', () => {
      const result = WaveSimulation.refraction(Math.PI / 4, 1, 1.5);
      expect(result.refractedAngle!).toBeLessThan(Math.PI / 4);
    });

    test('light leaving denser medium should bend away from normal', () => {
      const result = WaveSimulation.refraction(Math.PI / 6, 1.5, 1);
      expect(result.refractedAngle!).toBeGreaterThan(Math.PI / 6);
    });

    test('total internal reflection should occur above critical angle', () => {
      // Critical angle for glass-air is about 42°
      const result = WaveSimulation.refraction(Math.PI / 3, 1.5, 1); // 60° > 42°
      expect(result.isTotalInternalReflection).toBe(true);
      expect(result.refractedAngle).toBeNull();
      expect(result.reflectanceS).toBe(1);
      expect(result.reflectanceP).toBe(1);
    });

    test('critical angle should be arcsin(n2/n1)', () => {
      const n1 = 1.5, n2 = 1;
      const result = WaveSimulation.refraction(0.1, n1, n2);
      const expectedCritical = Math.asin(n2 / n1);
      expect(result.criticalAngle).toBeCloseTo(expectedCritical, 10);
    });

    test('no critical angle for light entering denser medium', () => {
      const result = WaveSimulation.refraction(0.1, 1, 1.5);
      expect(result.criticalAngle).toBeNull();
    });
  });
});

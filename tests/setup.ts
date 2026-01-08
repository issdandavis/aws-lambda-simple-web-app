/**
 * Jest Setup File
 */

// Set test environment variables
process.env.TABLE_NAME = 'test-physics-simulations';
process.env.BUCKET_NAME = 'test-physics-simulation-results';
process.env.STAGE = 'test';

// Mock AWS SDK
jest.mock('@aws-sdk/client-dynamodb', () => ({
  DynamoDBClient: jest.fn(() => ({})),
}));

jest.mock('@aws-sdk/lib-dynamodb', () => ({
  DynamoDBDocumentClient: {
    from: jest.fn(() => ({
      send: jest.fn().mockResolvedValue({}),
    })),
  },
  PutCommand: jest.fn(),
}));

jest.mock('@aws-sdk/client-s3', () => ({
  S3Client: jest.fn(() => ({
    send: jest.fn().mockResolvedValue({}),
  })),
  PutObjectCommand: jest.fn(),
}));

// Increase timeout for complex calculations
jest.setTimeout(10000);

// Custom matchers for physics tests
expect.extend({
  toBeCloseToPhysics(received: number, expected: number, precision: number = 1e-10) {
    const pass = Math.abs(received - expected) < Math.abs(expected) * precision + precision;
    if (pass) {
      return {
        message: () => `expected ${received} not to be close to ${expected} within ${precision * 100}%`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${received} to be close to ${expected} within ${precision * 100}%`,
        pass: false,
      };
    }
  },
});

declare global {
  namespace jest {
    interface Matchers<R> {
      toBeCloseToPhysics(expected: number, precision?: number): R;
    }
  }
}

export {};

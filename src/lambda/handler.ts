/**
 * AWS Lambda Handler for Physics Simulation API
 */

import { APIGatewayProxyEvent, APIGatewayProxyResult, Context } from 'aws-lambda';
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, PutCommand } from '@aws-sdk/lib-dynamodb';
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import { SimulationEngine } from './simulation-engine';
import { validateRequest, ValidationError } from './validator';
import { SimulationResult } from './types';

// Initialize clients
const dynamoClient = new DynamoDBClient({});
const docClient = DynamoDBDocumentClient.from(dynamoClient);
const s3Client = new S3Client({});

// Get environment variables
const TABLE_NAME = process.env.TABLE_NAME || 'physics-simulations';
const BUCKET_NAME = process.env.BUCKET_NAME || 'physics-simulation-results';

// Initialize simulation engine
const engine = new SimulationEngine();

/**
 * Create API response
 */
function createResponse(statusCode: number, body: unknown): APIGatewayProxyResult {
  return {
    statusCode,
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Content-Type,X-Api-Key,Authorization',
      'Access-Control-Allow-Methods': 'POST,OPTIONS',
    },
    body: JSON.stringify(body),
  };
}

/**
 * Save simulation metadata to DynamoDB
 */
async function saveMetadata(result: SimulationResult): Promise<void> {
  if (!result.metadata) return;

  try {
    await docClient.send(new PutCommand({
      TableName: TABLE_NAME,
      Item: {
        simulationId: result.metadata.simulationId,
        timestamp: result.metadata.timestamp,
        simulationType: result.simulationType,
        operation: result.operation,
        success: result.success,
        executionTimeMs: result.metadata.executionTimeMs,
        constantsUsed: result.metadata.constantsUsed,
        ttl: Math.floor(Date.now() / 1000) + 86400 * 30, // 30 days TTL
      },
    }));
  } catch (error) {
    console.error('Failed to save metadata to DynamoDB:', error);
    // Don't throw - this is a non-critical operation
  }
}

/**
 * Save simulation results to S3
 */
async function saveResults(result: SimulationResult): Promise<string | undefined> {
  if (!result.metadata || !result.success) return undefined;

  try {
    const key = `results/${result.simulationType}/${result.operation}/${result.metadata.simulationId}.json`;

    await s3Client.send(new PutObjectCommand({
      Bucket: BUCKET_NAME,
      Key: key,
      Body: JSON.stringify(result, null, 2),
      ContentType: 'application/json',
      Metadata: {
        'simulation-id': result.metadata.simulationId,
        'simulation-type': result.simulationType,
        'operation': result.operation,
      },
    }));

    return key;
  } catch (error) {
    console.error('Failed to save results to S3:', error);
    return undefined;
  }
}

/**
 * Main Lambda handler
 */
export async function handler(
  event: APIGatewayProxyEvent,
  context: Context
): Promise<APIGatewayProxyResult> {
  console.log('Request received:', {
    requestId: context.awsRequestId,
    path: event.path,
    method: event.httpMethod,
  });

  // Handle preflight requests
  if (event.httpMethod === 'OPTIONS') {
    return createResponse(200, { message: 'OK' });
  }

  // Only accept POST requests
  if (event.httpMethod !== 'POST') {
    return createResponse(405, {
      error: 'Method Not Allowed',
      message: 'Only POST requests are accepted',
    });
  }

  // Parse request body
  let body: unknown;
  try {
    body = JSON.parse(event.body || '{}');
  } catch {
    return createResponse(400, {
      error: 'Bad Request',
      message: 'Invalid JSON in request body',
    });
  }

  // Validate request
  let validatedRequest;
  try {
    validatedRequest = validateRequest(body);
  } catch (error) {
    if (error instanceof ValidationError) {
      return createResponse(400, {
        error: 'Validation Error',
        message: error.message,
        field: error.field,
      });
    }
    return createResponse(400, {
      error: 'Bad Request',
      message: error instanceof Error ? error.message : 'Unknown validation error',
    });
  }

  // Execute simulation
  const result = await engine.execute(validatedRequest);

  // Save metadata to DynamoDB
  await saveMetadata(result);

  // Save results to S3 if requested
  if (validatedRequest.options?.saveToS3 && result.success) {
    const s3Key = await saveResults(result);
    if (s3Key && result.metadata) {
      result.metadata.s3Key = s3Key;
    }
  }

  // Return result
  if (result.success) {
    return createResponse(200, result);
  } else {
    return createResponse(500, result);
  }
}

/**
 * Health check handler
 */
export async function healthHandler(): Promise<APIGatewayProxyResult> {
  return createResponse(200, {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: '1.0.0',
  });
}

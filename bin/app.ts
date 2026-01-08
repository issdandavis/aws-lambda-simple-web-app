#!/usr/bin/env node
/**
 * CDK App Entry Point
 */

import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { PhysicsSimulationStack } from '../lib/physics-simulation-stack';

const app = new cdk.App();

// Get stage from context or environment variable
const stage = app.node.tryGetContext('stage') || process.env.STAGE || 'dev';

// Validate stage
if (!['dev', 'staging', 'prod'].includes(stage)) {
  throw new Error(`Invalid stage: ${stage}. Must be one of: dev, staging, prod`);
}

// Create the stack
new PhysicsSimulationStack(app, `PhysicsSimulationStack-${stage}`, {
  stage: stage as 'dev' | 'staging' | 'prod',
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION || 'us-west-2',
  },
  tags: {
    Project: 'physics-simulation',
    Environment: stage,
    ManagedBy: 'CDK',
  },
});

app.synth();

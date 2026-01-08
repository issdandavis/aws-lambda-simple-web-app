/**
 * AWS CDK Stack for Physics Simulation Engine
 * Production-ready infrastructure with monitoring and security
 */

import * as cdk from 'aws-cdk-lib';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as apigateway from 'aws-cdk-lib/aws-apigateway';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import * as cloudwatch_actions from 'aws-cdk-lib/aws-cloudwatch-actions';
import * as sns from 'aws-cdk-lib/aws-sns';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';

export interface PhysicsSimulationStackProps extends cdk.StackProps {
  stage?: 'dev' | 'staging' | 'prod';
  alarmEmail?: string;
}

export class PhysicsSimulationStack extends cdk.Stack {
  public readonly apiEndpoint: string;
  public readonly apiKeyId: string;

  constructor(scope: Construct, id: string, props?: PhysicsSimulationStackProps) {
    super(scope, id, props);

    const stage = props?.stage || 'dev';
    const isProd = stage === 'prod';

    // ============================================
    // S3 Bucket for Simulation Results
    // ============================================
    const resultsBucket = new s3.Bucket(this, 'SimulationResultsBucket', {
      bucketName: `physics-simulation-results-${this.account}-${this.region}`,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      versioned: isProd,
      lifecycleRules: [
        {
          id: 'ExpireOldResults',
          enabled: true,
          expiration: cdk.Duration.days(isProd ? 365 : 30),
          transitions: isProd ? [
            {
              storageClass: s3.StorageClass.INFREQUENT_ACCESS,
              transitionAfter: cdk.Duration.days(30),
            },
            {
              storageClass: s3.StorageClass.GLACIER,
              transitionAfter: cdk.Duration.days(90),
            },
          ] : [],
        },
      ],
      removalPolicy: isProd ? cdk.RemovalPolicy.RETAIN : cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: !isProd,
    });

    // ============================================
    // DynamoDB Table for Simulation Metadata
    // ============================================
    const simulationTable = new dynamodb.Table(this, 'SimulationTable', {
      tableName: `physics-simulations-${stage}`,
      partitionKey: {
        name: 'simulationId',
        type: dynamodb.AttributeType.STRING,
      },
      sortKey: {
        name: 'timestamp',
        type: dynamodb.AttributeType.STRING,
      },
      billingMode: isProd
        ? dynamodb.BillingMode.PROVISIONED
        : dynamodb.BillingMode.PAY_PER_REQUEST,
      readCapacity: isProd ? 5 : undefined,
      writeCapacity: isProd ? 5 : undefined,
      timeToLiveAttribute: 'ttl',
      pointInTimeRecovery: isProd,
      encryption: dynamodb.TableEncryption.AWS_MANAGED,
      removalPolicy: isProd ? cdk.RemovalPolicy.RETAIN : cdk.RemovalPolicy.DESTROY,
    });

    // GSI for querying by simulation type
    simulationTable.addGlobalSecondaryIndex({
      indexName: 'SimulationTypeIndex',
      partitionKey: {
        name: 'simulationType',
        type: dynamodb.AttributeType.STRING,
      },
      sortKey: {
        name: 'timestamp',
        type: dynamodb.AttributeType.STRING,
      },
      projectionType: dynamodb.ProjectionType.ALL,
    });

    // ============================================
    // Lambda Function
    // ============================================
    const simulationLambda = new lambda.Function(this, 'SimulationLambda', {
      functionName: `physics-simulation-engine-${stage}`,
      runtime: lambda.Runtime.NODEJS_20_X,
      handler: 'src/lambda/handler.handler',
      code: lambda.Code.fromAsset('.', {
        exclude: [
          'cdk.out',
          'node_modules/.cache',
          'tests',
          '*.md',
          '.git',
          '.gitignore',
          'coverage',
        ],
        bundling: {
          image: lambda.Runtime.NODEJS_20_X.bundlingImage,
          command: [
            'bash', '-c', [
              'npm ci',
              'npm run build',
              'cp -r dist/* /asset-output/',
              'cp -r node_modules /asset-output/',
              'cp package.json /asset-output/',
            ].join(' && '),
          ],
        },
      }),
      memorySize: isProd ? 512 : 256,
      timeout: cdk.Duration.seconds(30),
      environment: {
        TABLE_NAME: simulationTable.tableName,
        BUCKET_NAME: resultsBucket.bucketName,
        STAGE: stage,
        NODE_OPTIONS: '--enable-source-maps',
      },
      logRetention: isProd ? logs.RetentionDays.ONE_YEAR : logs.RetentionDays.ONE_WEEK,
      tracing: lambda.Tracing.ACTIVE,
      reservedConcurrentExecutions: isProd ? 100 : 10,
    });

    // Grant permissions
    resultsBucket.grantReadWrite(simulationLambda);
    simulationTable.grantReadWriteData(simulationLambda);

    // Health check Lambda
    const healthLambda = new lambda.Function(this, 'HealthLambda', {
      functionName: `physics-simulation-health-${stage}`,
      runtime: lambda.Runtime.NODEJS_20_X,
      handler: 'src/lambda/handler.healthHandler',
      code: lambda.Code.fromAsset('.', {
        exclude: [
          'cdk.out',
          'node_modules/.cache',
          'tests',
          '*.md',
          '.git',
          '.gitignore',
          'coverage',
        ],
        bundling: {
          image: lambda.Runtime.NODEJS_20_X.bundlingImage,
          command: [
            'bash', '-c', [
              'npm ci',
              'npm run build',
              'cp -r dist/* /asset-output/',
              'cp -r node_modules /asset-output/',
              'cp package.json /asset-output/',
            ].join(' && '),
          ],
        },
      }),
      memorySize: 128,
      timeout: cdk.Duration.seconds(10),
      logRetention: logs.RetentionDays.ONE_WEEK,
    });

    // ============================================
    // API Gateway with API Key Authentication
    // ============================================
    const api = new apigateway.RestApi(this, 'PhysicsSimulationApi', {
      restApiName: `Physics Simulation API (${stage})`,
      description: 'Production-ready physics simulation API with quantum mechanics, particle dynamics, and wave simulations',
      deployOptions: {
        stageName: stage,
        throttlingBurstLimit: isProd ? 1000 : 100,
        throttlingRateLimit: isProd ? 500 : 50,
        metricsEnabled: true,
        loggingLevel: apigateway.MethodLoggingLevel.INFO,
        dataTraceEnabled: !isProd,
        tracingEnabled: true,
      },
      defaultCorsPreflightOptions: {
        allowOrigins: apigateway.Cors.ALL_ORIGINS,
        allowMethods: ['POST', 'OPTIONS'],
        allowHeaders: ['Content-Type', 'X-Api-Key', 'Authorization'],
      },
    });

    // API Key
    const apiKey = api.addApiKey('PhysicsApiKey', {
      apiKeyName: `physics-simulation-key-${stage}`,
      description: 'API key for Physics Simulation API',
    });

    // Usage Plan
    const usagePlan = api.addUsagePlan('UsagePlan', {
      name: `physics-simulation-usage-${stage}`,
      description: 'Usage plan for Physics Simulation API',
      throttle: {
        rateLimit: isProd ? 500 : 50,
        burstLimit: isProd ? 1000 : 100,
      },
      quota: {
        limit: isProd ? 100000 : 10000,
        period: apigateway.Period.MONTH,
      },
    });

    usagePlan.addApiKey(apiKey);
    usagePlan.addApiStage({
      stage: api.deploymentStage,
    });

    // Request validator
    const requestValidator = new apigateway.RequestValidator(this, 'RequestValidator', {
      restApi: api,
      requestValidatorName: 'validate-body',
      validateRequestBody: true,
      validateRequestParameters: true,
    });

    // Request model for simulation
    const simulationModel = api.addModel('SimulationRequestModel', {
      contentType: 'application/json',
      modelName: 'SimulationRequest',
      schema: {
        type: apigateway.JsonSchemaType.OBJECT,
        required: ['simulationType', 'operation', 'parameters'],
        properties: {
          simulationType: {
            type: apigateway.JsonSchemaType.STRING,
            enum: ['quantum', 'particle', 'wave', 'constants'],
          },
          operation: {
            type: apigateway.JsonSchemaType.STRING,
          },
          parameters: {
            type: apigateway.JsonSchemaType.OBJECT,
          },
          options: {
            type: apigateway.JsonSchemaType.OBJECT,
            properties: {
              saveToS3: { type: apigateway.JsonSchemaType.BOOLEAN },
              includeMetadata: { type: apigateway.JsonSchemaType.BOOLEAN },
              precision: {
                type: apigateway.JsonSchemaType.STRING,
                enum: ['standard', 'high'],
              },
            },
          },
        },
      },
    });

    // API Resources
    const simulateResource = api.root.addResource('simulate');
    const healthResource = api.root.addResource('health');

    // POST /simulate
    simulateResource.addMethod('POST', new apigateway.LambdaIntegration(simulationLambda, {
      proxy: true,
    }), {
      apiKeyRequired: true,
      requestValidator,
      requestModels: {
        'application/json': simulationModel,
      },
    });

    // GET /health
    healthResource.addMethod('GET', new apigateway.LambdaIntegration(healthLambda, {
      proxy: true,
    }), {
      apiKeyRequired: false,
    });

    // ============================================
    // CloudWatch Alarms and Monitoring
    // ============================================

    // SNS Topic for alarms
    const alarmTopic = new sns.Topic(this, 'AlarmTopic', {
      topicName: `physics-simulation-alarms-${stage}`,
      displayName: 'Physics Simulation Alarms',
    });

    // Lambda error rate alarm
    const errorAlarm = new cloudwatch.Alarm(this, 'LambdaErrorAlarm', {
      alarmName: `physics-simulation-errors-${stage}`,
      alarmDescription: 'Lambda function error rate exceeded threshold',
      metric: simulationLambda.metricErrors({
        period: cdk.Duration.minutes(5),
        statistic: 'Sum',
      }),
      threshold: isProd ? 10 : 50,
      evaluationPeriods: 2,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
    });
    errorAlarm.addAlarmAction(new cloudwatch_actions.SnsAction(alarmTopic));

    // Lambda duration alarm
    const durationAlarm = new cloudwatch.Alarm(this, 'LambdaDurationAlarm', {
      alarmName: `physics-simulation-duration-${stage}`,
      alarmDescription: 'Lambda function duration exceeded threshold',
      metric: simulationLambda.metricDuration({
        period: cdk.Duration.minutes(5),
        statistic: 'p95',
      }),
      threshold: 10000, // 10 seconds
      evaluationPeriods: 3,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
    });
    durationAlarm.addAlarmAction(new cloudwatch_actions.SnsAction(alarmTopic));

    // Lambda throttle alarm
    const throttleAlarm = new cloudwatch.Alarm(this, 'LambdaThrottleAlarm', {
      alarmName: `physics-simulation-throttles-${stage}`,
      alarmDescription: 'Lambda function throttled',
      metric: simulationLambda.metricThrottles({
        period: cdk.Duration.minutes(5),
        statistic: 'Sum',
      }),
      threshold: 1,
      evaluationPeriods: 1,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
    });
    throttleAlarm.addAlarmAction(new cloudwatch_actions.SnsAction(alarmTopic));

    // API Gateway 4xx alarm
    const api4xxAlarm = new cloudwatch.Alarm(this, 'Api4xxAlarm', {
      alarmName: `physics-simulation-4xx-${stage}`,
      alarmDescription: 'API Gateway 4xx error rate exceeded threshold',
      metric: api.metricClientError({
        period: cdk.Duration.minutes(5),
        statistic: 'Sum',
      }),
      threshold: isProd ? 100 : 500,
      evaluationPeriods: 2,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
    });
    api4xxAlarm.addAlarmAction(new cloudwatch_actions.SnsAction(alarmTopic));

    // API Gateway 5xx alarm
    const api5xxAlarm = new cloudwatch.Alarm(this, 'Api5xxAlarm', {
      alarmName: `physics-simulation-5xx-${stage}`,
      alarmDescription: 'API Gateway 5xx error rate exceeded threshold',
      metric: api.metricServerError({
        period: cdk.Duration.minutes(5),
        statistic: 'Sum',
      }),
      threshold: isProd ? 5 : 25,
      evaluationPeriods: 2,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
    });
    api5xxAlarm.addAlarmAction(new cloudwatch_actions.SnsAction(alarmTopic));

    // DynamoDB throttle alarm (for provisioned mode)
    if (isProd) {
      const dynamoThrottleAlarm = new cloudwatch.Alarm(this, 'DynamoThrottleAlarm', {
        alarmName: `physics-simulation-dynamo-throttle-${stage}`,
        alarmDescription: 'DynamoDB throttled requests',
        metric: simulationTable.metricThrottledRequestsForOperations({
          operations: [dynamodb.Operation.PUT_ITEM, dynamodb.Operation.GET_ITEM],
          period: cdk.Duration.minutes(5),
          statistic: 'Sum',
        }),
        threshold: 1,
        evaluationPeriods: 1,
        treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
        comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
      });
      dynamoThrottleAlarm.addAlarmAction(new cloudwatch_actions.SnsAction(alarmTopic));
    }

    // ============================================
    // CloudWatch Dashboard
    // ============================================
    const dashboard = new cloudwatch.Dashboard(this, 'SimulationDashboard', {
      dashboardName: `physics-simulation-${stage}`,
    });

    dashboard.addWidgets(
      new cloudwatch.TextWidget({
        markdown: `# Physics Simulation Engine Dashboard (${stage})`,
        width: 24,
        height: 1,
      })
    );

    dashboard.addWidgets(
      new cloudwatch.GraphWidget({
        title: 'Lambda Invocations',
        left: [simulationLambda.metricInvocations({ period: cdk.Duration.minutes(1) })],
        width: 8,
      }),
      new cloudwatch.GraphWidget({
        title: 'Lambda Duration',
        left: [
          simulationLambda.metricDuration({ period: cdk.Duration.minutes(1), statistic: 'p50' }),
          simulationLambda.metricDuration({ period: cdk.Duration.minutes(1), statistic: 'p95' }),
          simulationLambda.metricDuration({ period: cdk.Duration.minutes(1), statistic: 'p99' }),
        ],
        width: 8,
      }),
      new cloudwatch.GraphWidget({
        title: 'Lambda Errors',
        left: [simulationLambda.metricErrors({ period: cdk.Duration.minutes(1) })],
        width: 8,
      })
    );

    dashboard.addWidgets(
      new cloudwatch.GraphWidget({
        title: 'API Gateway Requests',
        left: [api.metricCount({ period: cdk.Duration.minutes(1) })],
        width: 8,
      }),
      new cloudwatch.GraphWidget({
        title: 'API Gateway Latency',
        left: [
          api.metricLatency({ period: cdk.Duration.minutes(1), statistic: 'p50' }),
          api.metricLatency({ period: cdk.Duration.minutes(1), statistic: 'p95' }),
        ],
        width: 8,
      }),
      new cloudwatch.GraphWidget({
        title: 'API Gateway Errors',
        left: [
          api.metricClientError({ period: cdk.Duration.minutes(1) }),
          api.metricServerError({ period: cdk.Duration.minutes(1) }),
        ],
        width: 8,
      })
    );

    dashboard.addWidgets(
      new cloudwatch.GraphWidget({
        title: 'DynamoDB Read/Write Units',
        left: [
          simulationTable.metricConsumedReadCapacityUnits({ period: cdk.Duration.minutes(1) }),
          simulationTable.metricConsumedWriteCapacityUnits({ period: cdk.Duration.minutes(1) }),
        ],
        width: 12,
      }),
      new cloudwatch.GraphWidget({
        title: 'S3 Bucket Size',
        left: [
          new cloudwatch.Metric({
            namespace: 'AWS/S3',
            metricName: 'BucketSizeBytes',
            dimensionsMap: {
              BucketName: resultsBucket.bucketName,
              StorageType: 'StandardStorage',
            },
            period: cdk.Duration.days(1),
            statistic: 'Average',
          }),
        ],
        width: 12,
      })
    );

    // ============================================
    // Outputs
    // ============================================
    this.apiEndpoint = api.url;
    this.apiKeyId = apiKey.keyId;

    new cdk.CfnOutput(this, 'ApiEndpoint', {
      value: api.url,
      description: 'API Gateway endpoint URL',
      exportName: `PhysicsSimulationApiEndpoint-${stage}`,
    });

    new cdk.CfnOutput(this, 'ApiKeyId', {
      value: apiKey.keyId,
      description: 'API Key ID (retrieve value from AWS Console)',
      exportName: `PhysicsSimulationApiKeyId-${stage}`,
    });

    new cdk.CfnOutput(this, 'SimulationTableName', {
      value: simulationTable.tableName,
      description: 'DynamoDB table name',
      exportName: `PhysicsSimulationTableName-${stage}`,
    });

    new cdk.CfnOutput(this, 'ResultsBucketName', {
      value: resultsBucket.bucketName,
      description: 'S3 bucket for simulation results',
      exportName: `PhysicsSimulationBucketName-${stage}`,
    });

    new cdk.CfnOutput(this, 'DashboardUrl', {
      value: `https://${this.region}.console.aws.amazon.com/cloudwatch/home?region=${this.region}#dashboards:name=${dashboard.dashboardName}`,
      description: 'CloudWatch Dashboard URL',
      exportName: `PhysicsSimulationDashboardUrl-${stage}`,
    });

    new cdk.CfnOutput(this, 'AlarmTopicArn', {
      value: alarmTopic.topicArn,
      description: 'SNS topic ARN for alarms',
      exportName: `PhysicsSimulationAlarmTopicArn-${stage}`,
    });
  }
}

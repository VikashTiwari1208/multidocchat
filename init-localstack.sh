#!/bin/bash
# Runs inside LocalStack on startup — creates all required AWS resources.
# awslocal is a LocalStack-aware wrapper around the AWS CLI.

set -e

echo "==> Creating S3 bucket..."
awslocal s3 mb s3://multidocchat-local

echo "==> Creating SQS queue..."
awslocal sqs create-queue --queue-name multidocchat-jobs

echo "==> Creating DynamoDB table..."
awslocal dynamodb create-table \
  --table-name multidocchat-sessions \
  --attribute-definitions AttributeName=session_id,AttributeType=S \
  --key-schema AttributeName=session_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1

echo "==> Enabling DynamoDB TTL on 'ttl' attribute..."
awslocal dynamodb update-time-to-live \
  --table-name multidocchat-sessions \
  --time-to-live-specification Enabled=true,AttributeName=ttl \
  --region us-east-1

echo "==> LocalStack resources ready."
echo "    S3 bucket : multidocchat-local"
echo "    SQS queue : multidocchat-jobs"
echo "    DynamoDB  : multidocchat-sessions"

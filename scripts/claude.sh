#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Bedrock auth/config; export these before running or inject here.
: "${AWS_BEARER_TOKEN_BEDROCK:?Set AWS_BEARER_TOKEN_BEDROCK}"
: "${AWS_DEFAULT_REGION:=us-east-2}"
: "${BEDROCK_INFERENCE_PROFILE:=arn:aws:bedrock:us-east-2:445527450773:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0}"

lib="${1:-torch}"          # torch|tf
llm="claude"

./scripts/llm_new_pipeline.sh "${lib}" "${llm}"

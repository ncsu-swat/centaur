#!/usr/bin/env python3
"""
Invoke the Anthropic Claude Sonnet 4.5 model on Bedrock using the bearer token flow.

Before running:
  export AWS_BEARER_TOKEN_BEDROCK=...   # required
  export AWS_DEFAULT_REGION=us-east-2   # optional (defaults to us-east-2)
  export BEDROCK_INFERENCE_PROFILE=arn:aws:bedrock:us-east-2:445527450773:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0
"""

import json
import os
import sys

import boto3


def main() -> None:
    prompt_text = " ".join(sys.argv[1:]).strip()
    if not prompt_text:
        prompt_text = "Hello from Codex CLI test. Please reply with a short confirmation."

    model_id = os.environ.get(
        "BEDROCK_INFERENCE_PROFILE",
        "arn:aws:bedrock:us-east-2:445527450773:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )

    client = boto3.client(
        "bedrock-runtime", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-2")
    )

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
            }
        ],
        "max_tokens": 256,
    }

    response = client.invoke_model(modelId=model_id, body=json.dumps(payload))
    print(response["body"].read().decode("utf-8"))


if __name__ == "__main__":
    main()

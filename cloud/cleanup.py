#!/usr/bin/env python3
"""
AWS SageMaker Endpoint Teardown Script
Author: Debojyoti Paul
Date: 2026-06-12

Deletes the real-time SageMaker GPU endpoint, endpoint configurations,
and registered model configurations to stop AWS hourly compute billing immediately.
"""

import sys
import argparse
import boto3
from botocore.exceptions import ClientError

def delete_endpoint(client, endpoint_name):
    """
    Deletes the active real-time SageMaker endpoint.
    This shuts down the underlying EC2 instances and stops billing immediately.
    """
    print(f"[*] Shutting down real-time endpoint '{endpoint_name}'...")
    try:
        client.delete_endpoint(EndpointName=endpoint_name)
        print(f"[+] Endpoint '{endpoint_name}' successfully deleted.")
        return True
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        # If the endpoint doesn't exist, log it and continue
        if error_code == "ValidationException" or "Could not find endpoint" in str(e):
            print(f"[Info] Endpoint '{endpoint_name}' is not currently active.")
            return True
        else:
            print(f"[Error] Failed to delete endpoint: {e}")
            return False

def delete_endpoint_config(client, endpoint_name):
    """
    Deletes the SageMaker endpoint configuration metadata.
    Config metadata does not accrue hourly charges but cleaning it up keeps the console tidy.
    """
    print(f"[*] Deleting endpoint configuration for '{endpoint_name}'...")
    try:
        client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        print(f"[+] Endpoint configuration deleted.")
        return True
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "ValidationException" or "Could not find" in str(e):
            print(f"[Info] Endpoint configuration '{endpoint_name}' does not exist.")
            return True
        else:
            print(f"[Error] Failed to delete endpoint config: {e}")
            return False

def delete_model(client, endpoint_name):
    """
    Deletes the registered model configuration.
    Model configuration maps S3 model artifacts to DL Container images in the registry.
    """
    print(f"[*] Deleting model registration configurations for '{endpoint_name}'...")
    try:
        client.delete_model(ModelName=endpoint_name)
        print(f"[+] Model registration deleted.")
        return True
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "ValidationException" or "Could not find" in str(e):
            print(f"[Info] Model registration '{endpoint_name}' does not exist.")
            return True
        else:
            print(f"[Error] Failed to delete model registration: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(
        description="Clean up AWS SageMaker real-time endpoints and model configurations"
    )
    parser.add_argument(
        "--endpoint-name",
        type=str,
        default="finqa-llama3-2-tuned-endpoint",
        help="Name of the SageMaker endpoint to delete"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS region where the endpoint is deployed"
    )
    args = parser.parse_args()

    print(f"[*] Initializing connection to SageMaker in region: {args.region}...")
    try:
        client = boto3.client("sagemaker", region_name=args.region)
    except Exception as e:
        print(f"[Error] Failed to initialize boto3 client: {e}")
        sys.exit(1)

    # Execute cleanup steps modularly
    endpoint_deleted = delete_endpoint(client, args.endpoint_name)
    config_deleted = delete_endpoint_config(client, args.endpoint_name)
    model_deleted = delete_model(client, args.endpoint_name)

    if endpoint_deleted and config_deleted and model_deleted:
        print("\n================================================================================")
        print("[+] CLEANUP COMPLETED! SageMaker billing has been successfully terminated.")
        print("================================================================================\n")
    else:
        print("\n================================================================================")
        print("[!] CLEANUP COMPLETED WITH ERRORS. Please verify remaining resources in AWS console.")
        print("================================================================================\n")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
AWS SageMaker LLM Serving Deployment Script
Author: Debojyoti Paul
Date: 2026-06-12

Handles packaging of the fused fine-tuned Safetensors weights,
uploading the archive to Amazon S3, and deploying a real-time endpoint
using SageMaker's Hugging Face Text Generation Inference (TGI) container.
"""

import os
import sys
import tarfile
import argparse
import boto3
import sagemaker
from sagemaker.huggingface import get_huggingface_llm_image_uri
from sagemaker.model import Model

# Configure project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def package_model(model_dir, output_tar_path):
    """
    Tars the fused model directory into model.tar.gz as required by SageMaker.
    """
    print(f"[*] Packaging model artifacts from '{model_dir}' into '{output_tar_path}'...")
    if not os.path.exists(model_dir):
        print(f"[Error] Source model directory '{model_dir}' does not exist.")
        sys.exit(1)
        
    with tarfile.open(output_tar_path, "w:gz") as tar:
        # Add all files in model_dir to root of the tar archive
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                filepath = os.path.join(root, file)
                # Keep archive paths relative to model_dir root
                arcname = os.path.relpath(filepath, model_dir)
                tar.add(filepath, arcname=arcname)
    print(f"[+] Model packaging complete. Size: {os.path.getsize(output_tar_path) / (1024*1024):.2f} MB")


def upload_model(sagemaker_session, tar_path, bucket):
    """
    Uploads the local model tarball archive to Amazon S3.
    """
    s3_key = "finqa-llama3.2-tuned/model.tar.gz"
    print(f"[*] Uploading model to S3: s3://{bucket}/{s3_key}...")
    s3_uri = sagemaker_session.upload_data(
        path=tar_path,
        bucket=bucket,
        key_prefix="finqa-llama3.2-tuned"
    )
    print(f"[+] Model successfully uploaded to S3: {s3_uri}")
    
    # Clean up local tar file to save space
    if os.path.exists(tar_path):
        os.remove(tar_path)
        print("[*] Cleaned up local model.tar.gz archive.")
        
    return s3_uri


def deploy_endpoint(sagemaker_session, image_uri, s3_uri, role, instance_type, region, hf_model_id=None, hf_token=None):
    """
    Registers the model config in SageMaker and deploys a real-time endpoint.
    """
    # Define SageMaker Model pointing to S3 or Hugging Face Hub
    endpoint_name = "finqa-llama3-2-tuned-endpoint"
    print(f"[*] Defining SageMaker Model configuration...")
    
    # Configure TGI Environment Variables
    env_config = {
        'HF_MODEL_ID': hf_model_id if hf_model_id else '/opt/ml/model',         # SageMaker automatically untars s3 data to /opt/ml/model
        'MAX_INPUT_LENGTH': '2048',             # Configured input context constraints
        'MAX_TOTAL_TOKENS': '4096',            # Target max capacity
        'MODEL_CARD': 'llama3.2-3b-instruct',
        'NUMBER_OF_GPU': '1'
    }

    if hf_token:
        env_config['HF_TOKEN'] = hf_token
        env_config['HUGGING_FACE_HUB_TOKEN'] = hf_token

    model = Model(
        image_uri=image_uri,
        model_data=s3_uri,
        role=role,
        env=env_config,
        sagemaker_session=sagemaker_session
    )

    # Deploy Endpoint
    print(f"[*] Deploying Real-Time Endpoint '{endpoint_name}' on instance '{instance_type}'...")
    print("[!] Warning: This process typically takes 5-8 minutes to allocate hardware. Do not close the terminal.")
    
    try:
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            container_startup_health_check_timeout=600 # Allow time for model loading
        )
        print("\n================================================================================")
        print("[+] DEPLOYMENT SUCCESSFUL!")
        print("================================================================================")
        print(f"Endpoint Name: {endpoint_name}")
        if s3_uri:
            print(f"S3 Model Artifact: {s3_uri}")
        else:
            print(f"Hugging Face Model ID: {hf_model_id}")
        print(f"Hardware Instance: {instance_type}")
        print("================================================================================\n")
        print("You can now run benchmarks pointing to SageMaker by updating your runner configuration.")
    except Exception as e:
        print(f"\n[Error] Deployment failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Deploy FinQA Tuned Model to SageMaker")
    parser.add_argument("--bucket", type=str, default=None, help="Target Amazon S3 bucket name (required if deploying local model)")
    parser.add_argument("--role", type=str, required=True, help="AWS IAM Execution Role ARN for SageMaker")
    parser.add_argument("--instance-type", type=str, default="ml.g4dn.xlarge", help="EC2 instance type (e.g., ml.g4dn.xlarge, ml.g5.xlarge)")
    parser.add_argument("--region", type=str, default="us-east-1", help="Target AWS region")
    parser.add_argument("--s3-uri", type=str, default=None, help="Skip packaging/uploading and use this S3 model URI directly")
    parser.add_argument("--skip-package", action="store_true", help="Skip packaging local fused_model directory into model.tar.gz")
    parser.add_argument("--hf-model-id", type=str, default=None, help="Hugging Face Hub model ID (e.g. Qwen/Qwen2.5-3B-Instruct or meta-llama/Llama-3.2-3B-Instruct)")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face API token for downloading gated models")
    parser.add_argument("--tgi-version", type=str, default="3.0.1", help="Hugging Face TGI container version to use on SageMaker")
    args = parser.parse_args()

    # Conditional validation: bucket is only required if packaging/uploading a local model
    if not args.hf_model_id and not args.s3_uri and not args.bucket:
        parser.error("--bucket is required unless deploying directly from Hugging Face Hub (--hf-model-id) or providing a pre-existing S3 model URI (--s3-uri)")

    model_dir = os.path.join(PROJECT_ROOT, "fused_model")
    tar_path = os.path.join(PROJECT_ROOT, "model.tar.gz")

    # Initialize SageMaker session
    print(f"[*] Initializing SageMaker session in region: {args.region}...")
    boto_session = boto3.Session(region_name=args.region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)

    # 1. Resolve S3 model URI or direct HF Hub model ID
    s3_uri = None
    if args.hf_model_id:
        print(f"[+] Deploying model directly from Hugging Face Hub: {args.hf_model_id}")
        if args.hf_token:
            print("[+] Using provided Hugging Face access token for authentication.")
    else:
        if args.s3_uri:
            s3_uri = args.s3_uri
            print(f"[+] Skipping packaging and upload. Using pre-existing model S3 URI: {s3_uri}")
        else:
            # Package local model
            if not args.skip_package:
                package_model(model_dir, tar_path)
            else:
                print("[*] Skipping model packaging step.")
                
            # Upload model to S3
            s3_uri = upload_model(sagemaker_session, tar_path, args.bucket)

    # 2. Retrieve Hugging Face LLM DLC URI
    print("[*] Retrieving Hugging Face LLM DLC image URI...")
    image_uri = get_huggingface_llm_image_uri(
        backend="huggingface",
        region=args.region,
        version=args.tgi_version
    )
    print(f"[+] DL Container URI: {image_uri}")

    # 3. Deploy Endpoint
    deploy_endpoint(
        sagemaker_session=sagemaker_session,
        image_uri=image_uri,
        s3_uri=s3_uri,
        role=args.role,
        instance_type=args.instance_type,
        region=args.region,
        hf_model_id=args.hf_model_id,
        hf_token=args.hf_token
    )

if __name__ == "__main__":
    main()

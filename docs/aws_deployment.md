# AWS Deployment Playbook

1. **Build & Push to ECR**
   - `aws ecr create-repository --repository-name multimodal-clinical-assistant`
   - `docker build -f docker/Dockerfile -t multimodal-clinical-assistant:latest .`
   - Tag and push to ECR.
2. **Provision compute**
   - ECS on EC2 with `g4dn.xlarge` for GPU inference.
3. **Runtime config**
   - Enable NVIDIA runtime and set `NVIDIA_VISIBLE_DEVICES=all`.
4. **Storage**
   - Store checkpoints and manifests in S3 (`s3://mcia-artifacts/...`).
5. **Observability**
   - CloudWatch logs + metric alarms (latency p95, 5xx, GPU utilization).

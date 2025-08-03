# Docker Training Integration Test Report
**Generated:** 2025-07-29 14:26:39
**Overall Status:** FAILED

## Summary
- **Total Tests:** 7
- **Passed:** 3
- **Failed:** 3
- **Warnings:** 1
- **Success Rate:** 42.9%

## Test Results

### docker_compose_config ❌
**Status:** FAILED

**Error:** Missing services in config: ['qlora-training', 'training-dev', 'training-monitor']

### dockerfile_training_stage ✅
**Status:** PASSED

**Message:** Dockerfile training stage is properly configured
**Stages Found:** 2
**Cuda Dependencies:** ['transformers']

### docker_image_build ❌
**Status:** FAILED

**Error:** Docker build failed: #0 building with "default" instance using docker driver

#1 [internal] load build definition from Dockerfile
#1 DONE 0.0s

#1 [internal] load build definition from Dockerfile
#1 transferring dockerfile: 7.52kB done
#1 DONE 0.0s

#2 [internal] load metadata for docker.io/nvidia/cuda:12.6.0-cudnn8-devel-ubuntu22.04
#2 ERROR: docker.io/nvidia/cuda:12.6.0-cudnn8-devel-ubuntu22.04: not found

#3 [internal] load metadata for docker.io/nvidia/cuda:12.6.0-cudnn8-runtime-ubuntu22.04
#3 CANCELED
------
 > [internal] load metadata for docker.io/nvidia/cuda:12.6.0-cudnn8-devel-ubuntu22.04:
------
Dockerfile:120
--------------------
 118 |     # === STAGE 5: Training environment with GPU support ===
 119 |     # DK002: Use pinned NVIDIA CUDA base for training reproducibility
 120 | >>> FROM nvidia/cuda:12.6.0-cudnn8-devel-ubuntu22.04 AS training-builder
 121 |     
 122 |     # DK007: Configure environment for optimal GPU training
--------------------
ERROR: failed to build: failed to solve: nvidia/cuda:12.6.0-cudnn8-devel-ubuntu22.04: failed to resolve source metadata for docker.io/nvidia/cuda:12.6.0-cudnn8-devel-ubuntu22.04: docker.io/nvidia/cuda:12.6.0-cudnn8-devel-ubuntu22.04: not found

### gpu_availability ⚠️
**Status:** WARNING

**Message:** GPU not available - training will run on CPU only
**Error:** Unable to find image 'nvidia/cuda:12.6.0-base-ubuntu22.04' locally
12.6.0-base-ubuntu22.04: Pulling from nvidia/cuda
3713021b0277: Pulling fs layer
46c9c54348df: Pulling fs layer
efc9014e2a4c: Pulling fs layer
67b3546b211d: Pulling fs layer
d339273dfb7f: Pulling fs layer
67b3546b211d: Waiting
d339273dfb7f: Waiting
46c9c54348df: Verifying Checksum
46c9c54348df: Download complete
67b3546b211d: Download complete
3713021b0277: Verifying Checksum
3713021b0277: Download complete
d339273dfb7f: Verifying Checksum
d339273dfb7f: Download complete
3713021b0277: Pull complete
46c9c54348df: Pull complete
efc9014e2a4c: Verifying Checksum
efc9014e2a4c: Download complete
efc9014e2a4c: Pull complete
67b3546b211d: Pull complete
d339273dfb7f: Pull complete
Digest: sha256:5dca947f477ec8ea91624447bdad7cc5f6a0dfe038ef53642c884bf5416478a6
Status: Downloaded newer image for nvidia/cuda:12.6.0-base-ubuntu22.04
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]

Run 'docker run --help' for more information

### volumes_and_directories ✅
**Status:** PASSED

**Message:** All required directories are available
**Created Directories:** []
**Total Directories:** 6

### training_config_files ✅
**Status:** PASSED

**Message:** All training configuration files are present
**Files Checked:** 3
**File Sizes:** {'training/qlora_finetune.py': 11155, 'training/qlora_config.py': 7291, 'scripts/run_training.sh': 19763}

### api_endpoints_integration ❌
**Status:** FAILED

**Error:** API integration test failed: name 'null' is not defined

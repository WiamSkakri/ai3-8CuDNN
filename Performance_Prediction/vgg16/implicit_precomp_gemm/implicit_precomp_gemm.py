"""
VGG16 Performance Prediction Test with cuDNN Implicit Precomp GEMM Algorithm

This test evaluates the Implicit Precomp GEMM algorithm on VGG16 architecture.
Implicit Precomp GEMM is an optimized variant that precomputes certain
transformations for improved performance on specific hardware configurations.
"""

import torch
import torchvision.models as tvm
import ai3
import time
import sys
import os

# Add parent directories to path for imports
# Note: This file is in implicit_precomp_gemm/ subdirectory, so go up 3 levels to project root
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../..')))
from bench import predict_show_time  # noqa: E402
from test import compare_tensors  # noqa: E402

# Configuration
BATCH_SIZE = 10
ALGORITHM = 'implicit precomp gemm'


def test_vgg16_implicit_precomp_gemm():
    """
    Test VGG16 with cuDNN Implicit Precomp GEMM algorithm.

    Implicit Precomp GEMM performs direct convolution with precomputed
    transformations, optimized for specific GPU architectures and
    workload characteristics.
    """
    print("\n" + "=" * 70)
    print(
        f"VGG16 Performance Prediction - cuDNN {ALGORITHM.upper()} Algorithm")
    print("=" * 70)

    # Check cuDNN availability
    if not ai3.using_cudnn():
        print("ERROR: cuDNN is not available. This test requires cuDNN.")
        print("Please build ai3 with USE_CUDNN flag enabled.")
        return False

    # Prepare input data
    input_data = torch.randn(BATCH_SIZE, 3, 224, 224)
    print(f"\nInput shape: {tuple(input_data.shape)}")
    print(f"Batch size: {BATCH_SIZE}")

    # Load VGG16 model
    print("\nLoading VGG16 with pretrained weights...")
    vgg16 = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT)
    vgg16.eval()

    with torch.inference_mode():
        # Baseline: PyTorch with cuDNN
        print("\n[BASELINE] Running PyTorch VGG16...")
        torch_out = predict_show_time(
            vgg16, input_data, 'PyTorch VGG16 (cuDNN baseline)')

        # Test with ai3 Implicit Precomp GEMM algorithm
        print(f"\n[AI3] Converting model to use '{ALGORITHM}' algorithm...")
        model = ai3.convert(vgg16, {'conv2d': ALGORITHM})

        print(f"[AI3] Running VGG16 with {ALGORITHM.upper()}...")
        implicit_precomp_out = predict_show_time(
            model, input_data, f'ai3 VGG16 ({ALGORITHM.upper()} cuDNN)')

        # Verify correctness
        print("\n[VERIFICATION] Checking output correctness...")
        compare_tensors(
            implicit_precomp_out, torch_out, f'{ALGORITHM.upper()} vs PyTorch',
            print_pass=True, atol=1e-4)

    print("\n" + "=" * 70)
    print(f"✓ VGG16 {ALGORITHM.upper()} Test Completed Successfully")
    print("=" * 70 + "\n")
    return True


if __name__ == '__main__':
    success = test_vgg16_implicit_precomp_gemm()
    sys.exit(0 if success else 1)

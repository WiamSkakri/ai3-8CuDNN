"""
GoogLeNet Performance Prediction Test with cuDNN Implicit GEMM Algorithm

This test evaluates the Implicit GEMM algorithm on GoogLeNet architecture.
Implicit GEMM is a direct convolution variant optimized for modern GPUs
with Tensor Cores, using automatic variant selection.
"""

import torch
import torchvision.models as tvm
import ai3
import time
import sys
import os

# Add parent directories to path for imports
# Note: This file is in implicit_gemm/ subdirectory, so go up 3 levels to project root
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../..')))
from bench import predict_show_time  # noqa: E402
from test import compare_tensors  # noqa: E402

# Configuration
BATCH_SIZE = 10
ALGORITHM = 'implicit gemm'


def test_googlenet_implicit_gemm():
    """
    Test GoogLeNet with cuDNN Implicit GEMM algorithm.

    Implicit GEMM performs direct convolution using implicit matrix
    multiplication, optimized for modern GPUs. Uses guess mode by
    default to automatically select the best variant.
    """
    print("\n" + "=" * 70)
    print(
        f"GoogLeNet Performance Prediction - cuDNN {ALGORITHM.upper()} Algorithm")
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

    # Load GoogLeNet model
    print("\nLoading GoogLeNet with pretrained weights...")
    googlenet = tvm.googlenet(weights=tvm.GoogLeNet_Weights.DEFAULT)
    googlenet.eval()

    with torch.inference_mode():
        # Baseline: PyTorch with cuDNN
        print("\n[BASELINE] Running PyTorch GoogLeNet...")
        torch_out = predict_show_time(
            googlenet, input_data, 'PyTorch GoogLeNet (cuDNN baseline)')

        # Test with ai3 Implicit GEMM algorithm
        print(f"\n[AI3] Converting model to use '{ALGORITHM}' algorithm...")
        model = ai3.convert(googlenet, {'conv2d': ALGORITHM})

        print(f"[AI3] Running GoogLeNet with {ALGORITHM.upper()}...")
        implicit_out = predict_show_time(
            model, input_data, f'ai3 GoogLeNet ({ALGORITHM.upper()} cuDNN)')

        # Verify correctness
        print("\n[VERIFICATION] Checking output correctness...")
        compare_tensors(
            implicit_out, torch_out, f'{ALGORITHM.upper()} vs PyTorch',
            print_pass=True, atol=1e-4)

    print("\n" + "=" * 70)
    print(f"✓ GoogLeNet {ALGORITHM.upper()} Test Completed Successfully")
    print("=" * 70 + "\n")
    return True


if __name__ == '__main__':
    success = test_googlenet_implicit_gemm()
    sys.exit(0 if success else 1)


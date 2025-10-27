"""
ResNet152 Performance Prediction Test with cuDNN GEMM Algorithm

This test evaluates the GEMM (General Matrix Multiplication) algorithm
on ResNet152 architecture. GEMM transforms convolution operations into
matrix multiplications, which are highly optimized on GPUs.
"""

import torch
import torchvision.models as tvm
import ai3
import time
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))
from bench import predict_show_time  # noqa: E402
from test import compare_tensors  # noqa: E402

# Configuration
BATCH_SIZE = 10
ALGORITHM = 'gemm'


def test_resnet152_gemm():
    """
    Test ResNet152 with cuDNN GEMM algorithm.

    GEMM (General Matrix Multiplication) is a standard approach that
    transforms convolution into matrix multiplication operations.
    """
    print("\n" + "=" * 70)
    print(
        f"ResNet152 Performance Prediction - cuDNN {ALGORITHM.upper()} Algorithm")
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

    # Load ResNet152 model
    print("\nLoading ResNet152 with pretrained weights...")
    resnet152 = tvm.resnet152(weights=tvm.ResNet152_Weights.DEFAULT)
    resnet152.eval()

    with torch.inference_mode():
        # Baseline: PyTorch with cuDNN
        print("\n[BASELINE] Running PyTorch ResNet152...")
        torch_out = predict_show_time(
            resnet152, input_data, 'PyTorch ResNet152 (cuDNN baseline)')

        # Test with ai3 GEMM algorithm
        print(f"\n[AI3] Converting model to use '{ALGORITHM}' algorithm...")
        model = ai3.convert(resnet152, {'conv2d': ALGORITHM})

        print(f"[AI3] Running ResNet152 with {ALGORITHM.upper()}...")
        gemm_out = predict_show_time(
            model, input_data, f'ai3 ResNet152 ({ALGORITHM.upper()} cuDNN)')

        # Verify correctness
        print("\n[VERIFICATION] Checking output correctness...")
        compare_tensors(
            gemm_out, torch_out, f'{ALGORITHM.upper()} vs PyTorch',
            print_pass=True, atol=1e-4)

    print("\n" + "=" * 70)
    print(f"✓ ResNet152 {ALGORITHM.upper()} Test Completed Successfully")
    print("=" * 70 + "\n")
    return True


if __name__ == '__main__':
    success = test_resnet152_gemm()
    sys.exit(0 if success else 1)


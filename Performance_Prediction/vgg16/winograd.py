"""
VGG16 Performance Prediction Test with cuDNN Winograd Algorithm

This test evaluates the Winograd algorithm on VGG16 architecture.
Winograd uses minimal filtering algorithm for efficient convolution.

NOTE: Winograd has strict requirements:
- Only works with 3x3 kernels
- Only works with stride=1
- Both kernel height and width must be 3

VGG16 contains layers that meet these requirements, but also some that don't.
This test will attempt to use Winograd but may fail on incompatible layers.
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
ALGORITHM = 'winograd'


def test_vgg16_winograd():
    """
    Test VGG16 with cuDNN Winograd algorithm.

    Winograd uses the Winograd minimal filtering algorithm for efficient
    small convolutions. It only works with 3x3 kernels and stride=1.

    VGG16 uses 3x3 kernels throughout, making it a good candidate for
    Winograd testing.
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

    print("\n[INFO] Winograd Requirements:")
    print("  - Kernel size: 3x3 only")
    print("  - Stride: 1 only")
    print("  VGG16 uses 3x3 kernels with stride=1, so it's compatible!")

    # Load VGG16 model
    print("\nLoading VGG16 with pretrained weights...")
    vgg16 = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT)
    vgg16.eval()

    try:
        with torch.inference_mode():
            # Baseline: PyTorch with cuDNN
            print("\n[BASELINE] Running PyTorch VGG16...")
            torch_out = predict_show_time(
                vgg16, input_data, 'PyTorch VGG16 (cuDNN baseline)')

            # Test with ai3 Winograd algorithm
            print(
                f"\n[AI3] Converting model to use '{ALGORITHM}' algorithm...")
            model = ai3.convert(vgg16, {'conv2d': ALGORITHM})

            print(f"[AI3] Running VGG16 with {ALGORITHM.upper()}...")
            winograd_out = predict_show_time(
                model, input_data, f'ai3 VGG16 ({ALGORITHM.upper()} cuDNN)')

            # Verify correctness
            print("\n[VERIFICATION] Checking output correctness...")
            compare_tensors(
                winograd_out, torch_out, f'{ALGORITHM.upper()} vs PyTorch',
                print_pass=True, atol=1e-4)

        print("\n" + "=" * 70)
        print(f"✓ VGG16 {ALGORITHM.upper()} Test Completed Successfully")
        print("=" * 70 + "\n")
        return True

    except Exception as e:
        print(f"\n[ERROR] Test failed: {str(e)}")
        print("\nThis may occur if VGG16 has layers incompatible with Winograd.")
        print("Winograd only supports 3x3 kernels with stride=1.")
        print("=" * 70 + "\n")
        return False


if __name__ == '__main__':
    success = test_vgg16_winograd()
    sys.exit(0 if success else 1)

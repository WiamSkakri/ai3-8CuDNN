"""
DenseNet121 Performance Prediction Test with cuDNN FFT Algorithm

This test evaluates the FFT convolution algorithm on DenseNet121 architecture.
FFT convolution uses Fast Fourier Transform to perform convolution operations
in the frequency domain, which can be more efficient for larger kernel sizes.
"""

import torch
import torchvision.models as tvm
import ai3
import time
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../..')))
from bench import predict_show_time  # noqa: E402
from test import compare_tensors  # noqa: E402

# Configuration
BATCH_SIZE = 10
ALGORITHM = 'fft'


def test_densenet_fft():
    """
    Test DenseNet121 with cuDNN FFT convolution algorithm.

    FFT convolution transforms the convolution operation to the frequency
    domain using Fast Fourier Transform, which can provide performance
    benefits for certain input sizes and kernel configurations.
    """
    print("\n" + "=" * 70)
    print(
        f"DenseNet121 Performance Prediction - cuDNN {ALGORITHM.upper()} Algorithm")
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

    # Load DenseNet121 model
    print("\nLoading DenseNet121 with pretrained weights...")
    densenet = tvm.densenet121(weights=tvm.DenseNet121_Weights.DEFAULT)
    densenet.eval()

    with torch.inference_mode():
        # Baseline: PyTorch with cuDNN
        print("\n[BASELINE] Running PyTorch DenseNet121...")
        torch_out = predict_show_time(
            densenet, input_data, 'PyTorch DenseNet121 (cuDNN baseline)')

        # Test with ai3 FFT algorithm
        print(f"\n[AI3] Converting model to use '{ALGORITHM}' algorithm...")
        model = ai3.convert(densenet, {'conv2d': ALGORITHM})

        print(f"[AI3] Running DenseNet121 with {ALGORITHM.upper()}...")
        fft_out = predict_show_time(
            model, input_data, f'ai3 DenseNet121 ({ALGORITHM.upper()} cuDNN)')

        # Verify correctness
        print("\n[VERIFICATION] Checking output correctness...")
        compare_tensors(
            fft_out, torch_out, f'{ALGORITHM.upper()} vs PyTorch',
            print_pass=True, atol=1e-4)

    print("\n" + "=" * 70)
    print(f"✓ DenseNet121 {ALGORITHM.upper()} Test Completed Successfully")
    print("=" * 70 + "\n")
    return True


if __name__ == '__main__':
    success = test_densenet_fft()
    sys.exit(0 if success else 1)

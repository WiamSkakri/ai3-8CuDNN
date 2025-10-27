"""
GoogLeNet Performance Prediction Test with cuDNN FFT Algorithm
"""

import torch
import torchvision.models as tvm
import ai3
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../..')))
from bench import predict_show_time  # noqa: E402
from test import compare_tensors  # noqa: E402

BATCH_SIZE = 10
ALGORITHM = 'fft'


def test_googlenet_fft():
    print("\n" + "=" * 70)
    print(
        f"GoogLeNet Performance Prediction - cuDNN {ALGORITHM.upper()} Algorithm")
    print("=" * 70)

    if not ai3.using_cudnn():
        print("ERROR: cuDNN is not available.")
        return False

    input_data = torch.randn(BATCH_SIZE, 3, 224, 224)
    print(f"\nInput shape: {tuple(input_data.shape)}")

    print("\nLoading GoogLeNet...")
    googlenet = tvm.googlenet(weights=tvm.GoogLeNet_Weights.DEFAULT)
    googlenet.eval()

    with torch.inference_mode():
        print("\n[BASELINE] Running PyTorch GoogLeNet...")
        torch_out = predict_show_time(
            googlenet, input_data, 'PyTorch GoogLeNet')

        print(f"\n[AI3] Converting to '{ALGORITHM}'...")
        model = ai3.convert(googlenet, {'conv2d': ALGORITHM})

        print(f"[AI3] Running GoogLeNet with {ALGORITHM.upper()}...")
        fft_out = predict_show_time(
            model, input_data, f'ai3 GoogLeNet ({ALGORITHM.upper()})')

        print("\n[VERIFICATION] Checking correctness...")
        compare_tensors(
            fft_out, torch_out, f'{ALGORITHM.upper()} vs PyTorch',
            print_pass=True, atol=1e-4)

    print("\n" + "=" * 70)
    print(f"✓ GoogLeNet {ALGORITHM.upper()} Test Completed")
    print("=" * 70 + "\n")
    return True


if __name__ == '__main__':
    success = test_googlenet_fft()
    sys.exit(0 if success else 1)

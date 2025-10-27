"""
GoogLeNet Performance Prediction Test with cuDNN Direct Algorithm

This test evaluates the Direct convolution algorithm on GoogLeNet architecture.
"""

import torch
import torchvision.models as tvm
import ai3
import time
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../..')))
from bench import predict_show_time  # noqa: E402
from test import compare_tensors  # noqa: E402

BATCH_SIZE = 10
ALGORITHM = 'direct'


def test_googlenet_direct():
    print("\n" + "=" * 70)
    print(
        f"GoogLeNet Performance Prediction - cuDNN {ALGORITHM.upper()} Algorithm")
    print("=" * 70)

    if not ai3.using_cudnn():
        print("ERROR: cuDNN is not available. This test requires cuDNN.")
        return False

    input_data = torch.randn(BATCH_SIZE, 3, 224, 224)
    print(f"\nInput shape: {tuple(input_data.shape)}")

    print("\nLoading GoogLeNet with pretrained weights...")
    googlenet = tvm.googlenet(weights=tvm.GoogLeNet_Weights.DEFAULT)
    googlenet.eval()

    with torch.inference_mode():
        print("\n[BASELINE] Running PyTorch GoogLeNet...")
        torch_out = predict_show_time(
            googlenet, input_data, 'PyTorch GoogLeNet (cuDNN baseline)')

        print(f"\n[AI3] Converting model to use '{ALGORITHM}' algorithm...")
        model = ai3.convert(googlenet, {'conv2d': ALGORITHM})

        print(f"[AI3] Running GoogLeNet with {ALGORITHM.UPPER()}...")
        direct_out = predict_show_time(
            model, input_data, f'ai3 GoogLeNet ({ALGORITHM.upper()} cuDNN)')

        print("\n[VERIFICATION] Checking output correctness...")
        compare_tensors(
            direct_out, torch_out, f'{ALGORITHM.upper()} vs PyTorch',
            print_pass=True, atol=1e-4)

    print("\n" + "=" * 70)
    print(f"✓ GoogLeNet {ALGORITHM.upper()} Test Completed Successfully")
    print("=" * 70 + "\n")
    return True


if __name__ == '__main__':
    success = test_googlenet_direct()
    sys.exit(0 if success else 1)

"""GoogLeNet Performance Test with cuDNN FFT Tiling Algorithm"""

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
ALGORITHM = 'fft tiling'


def test_googlenet_fft_tiling():
    print("\n" + "=" * 70)
    print(f"GoogLeNet - cuDNN {ALGORITHM.upper()}")
    print("=" * 70)

    if not ai3.using_cudnn():
        print("ERROR: cuDNN not available")
        return False

    input_data = torch.randn(BATCH_SIZE, 3, 224, 224)
    googlenet = tvm.googlenet(weights=tvm.GoogLeNet_Weights.DEFAULT)
    googlenet.eval()

    with torch.inference_mode():
        torch_out = predict_show_time(
            googlenet, input_data, 'PyTorch GoogLeNet')
        model = ai3.convert(googlenet, {'conv2d': ALGORITHM})
        fft_tiling_out = predict_show_time(
            model, input_data, f'ai3 GoogLeNet ({ALGORITHM.upper()})')
        compare_tensors(fft_tiling_out, torch_out,
                        f'{ALGORITHM.upper()} vs PyTorch', print_pass=True, atol=1e-4)

    print(f"\n✓ GoogLeNet {ALGORITHM.upper()} Test Completed\n")
    return True


if __name__ == '__main__':
    sys.exit(0 if test_googlenet_fft_tiling() else 1)

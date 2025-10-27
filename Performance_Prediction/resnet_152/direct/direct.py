"""ResNet152 Performance Test with cuDNN Direct Algorithm"""

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
ALGORITHM = 'direct'


def test_resnet152_direct():
    print("\n" + "=" * 70)
    print(f"ResNet152 - cuDNN {ALGORITHM.upper()}")
    print("=" * 70)

    if not ai3.using_cudnn():
        print("ERROR: cuDNN not available")
        return False

    input_data = torch.randn(BATCH_SIZE, 3, 224, 224)
    resnet152 = tvm.resnet152(weights=tvm.ResNet152_Weights.DEFAULT)
    resnet152.eval()

    with torch.inference_mode():
        torch_out = predict_show_time(
            resnet152, input_data, 'PyTorch ResNet152')
        model = ai3.convert(resnet152, {'conv2d': ALGORITHM})
        direct_out = predict_show_time(
            model, input_data, f'ai3 ResNet152 ({ALGORITHM.upper()})')
        compare_tensors(direct_out, torch_out,
                        f'{ALGORITHM.upper()} vs PyTorch', print_pass=True, atol=1e-4)

    print(f"\n✓ ResNet152 {ALGORITHM.upper()} Test Completed\n")
    return True


if __name__ == '__main__':
    sys.exit(0 if test_resnet152_direct() else 1)

#!/usr/bin/env python3
"""Verification: PyTorch vs ai3 Direct (ResNet152)"""

import torch
import torchvision.models as models
import ai3
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../..')))

BATCH_SIZE, INPUT_SIZE, WARMUP, MEASURE = 1, 224, 5, 10


def measure_time(model, input_data, use_cuda=True):
    with torch.inference_mode():
        for _ in range(WARMUP):
            _ = model(input_data)
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

    times = []
    with torch.inference_mode():
        for _ in range(MEASURE):
            if use_cuda and torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(input_data)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            else:
                start_time = time.time()
                _ = model(input_data)
                times.append((time.time() - start_time) * 1000)

    return {'mean': np.mean(times), 'std': np.std(times)}


def main():
    print("="*80)
    print("VERIFICATION: PyTorch vs ai3 Direct (ResNet152)")
    print("="*80)

    use_cuda = torch.cuda.is_available()
    if not ai3.using_cudnn():
        print("✗ cuDNN not available")
        return False

    input_data = torch.randn(BATCH_SIZE, 3, INPUT_SIZE, INPUT_SIZE)
    resnet152 = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    resnet152.eval()

    pytorch_stats = measure_time(resnet152, input_data, use_cuda)
    print(f"PyTorch: {pytorch_stats['mean']:.2f}ms")

    ai3_model = ai3.swap_conv2d(resnet152, 'direct')
    ai3_stats = measure_time(ai3_model, input_data, use_cuda)
    print(f"ai3 Direct: {ai3_stats['mean']:.2f}ms")

    diff_pct = ((ai3_stats['mean'] - pytorch_stats['mean']
                 ) / pytorch_stats['mean']) * 100
    print(f"Difference: {diff_pct:+.1f}%")
    print("✓ Complete\n")
    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 1)

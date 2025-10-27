#!/usr/bin/env python3
"""Verification Script: PyTorch vs ai3 FFT (GoogLeNet)"""

import torch
import torchvision.models as models
import ai3
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../..')))

BATCH_SIZE = 1
INPUT_SIZE = 224
WARMUP = 5
MEASURE = 10


def measure_time(model, input_data, name, use_cuda=True):
    with torch.inference_mode():
        for _ in range(WARMUP):
            _ = model(input_data)
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

    times = []
    with torch.inference_mode():
        if use_cuda and torch.cuda.is_available():
            for _ in range(MEASURE):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(input_data)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
        else:
            for _ in range(MEASURE):
                start_time = time.time()
                _ = model(input_data)
                times.append((time.time() - start_time) * 1000)

    return {'mean': np.mean(times), 'std': np.std(times)}


def main():
    print("="*80)
    print("VERIFICATION: PyTorch vs ai3 FFT (GoogLeNet)")
    print("="*80)

    use_cuda = torch.cuda.is_available()
    print(
        f"✓ CUDA available: {torch.cuda.get_device_name(0) if use_cuda else 'No'}")

    if not ai3.using_cudnn():
        print("✗ ai3 cuDNN not available")
        return False
    print("✓ ai3 cuDNN confirmed")

    input_data = torch.randn(BATCH_SIZE, 3, INPUT_SIZE, INPUT_SIZE)
    googlenet = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
    googlenet.eval()

    pytorch_stats = measure_time(googlenet, input_data, "PyTorch", use_cuda)
    print(
        f"PyTorch: {pytorch_stats['mean']:.2f}ms ± {pytorch_stats['std']:.2f}ms")

    ai3_model = ai3.swap_conv2d(googlenet, 'fft')
    ai3_stats = measure_time(ai3_model, input_data, "ai3 FFT", use_cuda)
    print(f"ai3 FFT: {ai3_stats['mean']:.2f}ms ± {ai3_stats['std']:.2f}ms")

    diff_pct = ((ai3_stats['mean'] - pytorch_stats['mean']
                 ) / pytorch_stats['mean']) * 100
    print(f"Difference: {diff_pct:+.1f}%")
    print("✓ Verification complete\n")
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

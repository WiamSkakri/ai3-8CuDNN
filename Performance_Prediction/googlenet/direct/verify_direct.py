#!/usr/bin/env python3
"""
Verification Script: Compare PyTorch Default vs ai3 Direct (GoogLeNet)
"""

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
                end_time = time.time()
                times.append((end_time - start_time) * 1000)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }


def inspect_model_layers(model, name):
    print(f"\n{name} Layer Types:")
    ai3_count = 0
    pytorch_count = 0

    for layer_name, module in model.named_modules():
        if 'conv' in layer_name.lower():
            if hasattr(module, 'algorithm'):
                ai3_count += 1
                if ai3_count <= 5:
                    print(
                        f"  {layer_name}: ai3.Conv2D (algorithm={module.algorithm})")
            elif isinstance(module, torch.nn.Conv2d):
                pytorch_count += 1
                if pytorch_count <= 5:
                    print(f"  {layer_name}: torch.nn.Conv2d")

    print(f"  Summary: {ai3_count} ai3 layers, {pytorch_count} PyTorch layers")
    return ai3_count, pytorch_count


def main():
    print("="*80)
    print("VERIFICATION: PyTorch Default vs ai3 Direct (GoogLeNet)")
    print("="*80)

    if not torch.cuda.is_available():
        print("✗ CUDA not available - running on CPU")
        use_cuda = False
    else:
        use_cuda = True
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")

    if not ai3.using_cudnn():
        print("✗ ai3 cuDNN not available")
        return False
    print("✓ ai3 cuDNN support confirmed")

    input_data = torch.randn(BATCH_SIZE, 3, INPUT_SIZE, INPUT_SIZE)
    print(f"\nInput shape: {tuple(input_data.shape)}")

    print("\nLoading GoogLeNet...")
    googlenet = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
    googlenet.eval()

    print("\n" + "="*80)
    print("TEST 1: PyTorch GoogLeNet (Default cuDNN)")
    print("="*80)
    inspect_model_layers(googlenet, "PyTorch")
    pytorch_stats = measure_time(googlenet, input_data, "PyTorch", use_cuda)
    print(
        f"\nPerformance: {pytorch_stats['mean']:.2f}ms ± {pytorch_stats['std']:.2f}ms")

    print("\n" + "="*80)
    print("TEST 2: ai3 GoogLeNet (Explicit Direct)")
    print("="*80)
    print("Converting to ai3 with 'direct' algorithm...")
    ai3_model = ai3.swap_conv2d(googlenet, 'direct')
    ai3_count, pytorch_count = inspect_model_layers(ai3_model, "ai3")

    if ai3_count == 0:
        print("✗ ERROR: No ai3 layers found!")
        return False

    ai3_stats = measure_time(ai3_model, input_data, "ai3 Direct", use_cuda)
    print(
        f"\nPerformance: {ai3_stats['mean']:.2f}ms ± {ai3_stats['std']:.2f}ms")

    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(
        f"PyTorch (default):  {pytorch_stats['mean']:.2f}ms ± {pytorch_stats['std']:.2f}ms")
    print(
        f"ai3 (direct):       {ai3_stats['mean']:.2f}ms ± {ai3_stats['std']:.2f}ms")

    diff_pct = ((ai3_stats['mean'] - pytorch_stats['mean']
                 ) / pytorch_stats['mean']) * 100
    print(f"Difference:         {diff_pct:+.1f}%")

    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    print(
        f"✓ ai3 Direct algorithm successfully applied ({ai3_count} layers converted)")
    print("="*80 + "\n")
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

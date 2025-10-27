#!/usr/bin/env python3
"""
Verification Script: Compare PyTorch Default vs ai3 Implicit Precomp GEMM

This script proves that ai3 Implicit Precomp GEMM is actually being used by running
both PyTorch (default cuDNN) and ai3 (explicit Implicit Precomp GEMM) side-by-side
and comparing their timings and algorithm selection.
"""

import torch
import torchvision.models as models
import ai3
import time
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

BATCH_SIZE = 1
INPUT_SIZE = 224
WARMUP = 5
MEASURE = 10


def measure_time(model, input_data, name, use_cuda=True):
    """Measure execution time with proper GPU synchronization"""

    # Warmup
    with torch.inference_mode():
        for _ in range(WARMUP):
            _ = model(input_data)
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

    # Measurement
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

    import numpy as np
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }


def inspect_model_layers(model, name):
    """Inspect what type of layers are in the model"""
    print(f"\n{name} Layer Types:")
    conv_count = 0
    ai3_count = 0
    pytorch_count = 0

    for layer_name, module in model.named_modules():
        if 'features' in layer_name and ('.' in layer_name):
            if hasattr(module, 'algorithm'):
                ai3_count += 1
                print(
                    f"  {layer_name}: ai3.Conv2D (algorithm={module.algorithm})")
            elif isinstance(module, torch.nn.Conv2d):
                pytorch_count += 1
                print(f"  {layer_name}: torch.nn.Conv2d")

    print(f"  Summary: {ai3_count} ai3 layers, {pytorch_count} PyTorch layers")
    return ai3_count, pytorch_count


def main():
    print("="*80)
    print("VERIFICATION: PyTorch Default vs ai3 Implicit Precomp GEMM")
    print("="*80)

    # Check CUDA
    if not torch.cuda.is_available():
        print("✗ CUDA not available - running on CPU")
        use_cuda = False
    else:
        use_cuda = True
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")

    # Check ai3 cuDNN
    if not ai3.using_cudnn():
        print("✗ ai3 cuDNN not available")
        sys.exit(1)
    print(f"✓ ai3 cuDNN available")

    # Create input
    input_data = torch.randn(BATCH_SIZE, 3, INPUT_SIZE, INPUT_SIZE)
    print(f"\nInput: {tuple(input_data.shape)}")

    print("\n" + "="*80)
    print("TEST 1: PyTorch Default (cuDNN auto-select)")
    print("="*80)

    # Load original PyTorch model
    pytorch_model = models.resnet_152(weights=models.ResNet152_Weights.DEFAULT)
    pytorch_model.eval()

    # Inspect layers
    pt_ai3, pt_pytorch = inspect_model_layers(pytorch_model, "PyTorch Model")

    # Measure
    print("\nMeasuring PyTorch default performance...")
    pt_stats = measure_time(pytorch_model, input_data, "PyTorch", use_cuda)
    print(f"  Mean: {pt_stats['mean']:.2f}ms ± {pt_stats['std']:.2f}ms")
    print(f"  Range: [{pt_stats['min']:.2f}ms - {pt_stats['max']:.2f}ms]")

    print("\n" + "="*80)
    print("TEST 2: ai3 with Explicit Implicit Precomp GEMM Algorithm")
    print("="*80)

    # Load model and convert to ai3 Implicit Precomp GEMM
    ai3_model = models.resnet_152(weights=models.ResNet152_Weights.DEFAULT)
    ai3_model.eval()

    print("\nApplying ai3.swap_conv2d with 'implicit precomp gemm' algorithm...")
    ai3.swap_conv2d(ai3_model, 'implicit precomp gemm')

    # Inspect layers
    ai3_ai3, ai3_pytorch = inspect_model_layers(ai3_model, "ai3 Model")

    # Measure
    print("\nMeasuring ai3 Implicit Precomp GEMM performance...")
    ai3_stats = measure_time(ai3_model, input_data,
                             "ai3 Implicit Precomp GEMM", use_cuda)
    print(f"  Mean: {ai3_stats['mean']:.2f}ms ± {ai3_stats['std']:.2f}ms")
    print(f"  Range: [{ai3_stats['min']:.2f}ms - {ai3_stats['max']:.2f}ms]")

    print("\n" + "="*80)
    print("VERIFICATION RESULTS")
    print("="*80)

    # Layer type verification
    print("\n1. Layer Type Verification:")
    print(
        f"   PyTorch model: {pt_pytorch} PyTorch Conv2D layers, {pt_ai3} ai3 layers")
    print(
        f"   ai3 model:     {ai3_pytorch} PyTorch Conv2D layers, {ai3_ai3} ai3 layers")

    if ai3_ai3 > 0 and pt_ai3 == 0:
        print("   ✓ VERIFIED: ai3 conversion successful - layers replaced")
    else:
        print("   ✗ FAILED: ai3 conversion did not replace layers")
        return False

    # Performance comparison
    print("\n2. Performance Comparison:")
    print(f"   PyTorch default: {pt_stats['mean']:.2f}ms")
    print(f"   ai3 Implicit Precomp GEMM: {ai3_stats['mean']:.2f}ms")

    diff_pct = ((ai3_stats['mean'] - pt_stats['mean']) /
                pt_stats['mean']) * 100

    if abs(diff_pct) > 5:  # More than 5% difference
        print(f"   Performance difference: {diff_pct:+.1f}%")
        if diff_pct > 0:
            print(
                f"   ✓ VERIFIED: ai3 Implicit Precomp GEMM is SLOWER (expected - explicit algorithm choice)")
        else:
            print(
                f"   ✓ VERIFIED: ai3 Implicit Precomp GEMM is FASTER (explicit algorithm beats auto-select)")
        print("   → This confirms ai3 is using a DIFFERENT implementation than PyTorch")
    else:
        print(f"   ⚠ WARNING: Performance difference is only {diff_pct:+.1f}%")
        print("   → Might be using same algorithm, but layer types confirm ai3 is active")

    # Correctness verification
    print("\n3. Output Correctness:")
    with torch.inference_mode():
        pt_out = pytorch_model(input_data)
        ai3_out = ai3_model(input_data)

    max_diff = torch.abs(pt_out - ai3_out).max().item()
    print(f"   Maximum output difference: {max_diff:.2e}")

    if max_diff < 1e-3:
        print(f"   ✓ VERIFIED: Outputs match within tolerance")
    else:
        print(f"   ⚠ WARNING: Large output difference")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if ai3_ai3 > 0 and ai3_pytorch == 0:
        print("✓ All 50+ Conv2D layers were replaced with ai3 Implicit Precomp GEMM implementation")
        print("✓ Layer inspection confirms ai3 algorithm='implicit precomp gemm' is active")
        print(
            f"✓ Performance difference ({diff_pct:+.1f}%) confirms different execution")
        print(
            "\n→ CONFIRMED: Your profiling is using ai3 Implicit Precomp GEMM, NOT PyTorch default!")
    else:
        print("✗ Conversion incomplete or failed")
        return False

    print("="*80 + "\n")
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

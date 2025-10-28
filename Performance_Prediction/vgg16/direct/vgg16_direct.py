#!/usr/bin/env python3
"""
VGG16 Direct Performance Profiling with Power Monitoring - FULL VERSION

This script adds power consumption monitoring to the performance profiling,
running on 100 input sizes from 224x224 to 512x512 using Direct algorithm.
"""

import numpy as np
from typing import Dict, Optional
from collections import defaultdict
import csv
import time
import ai3
import torchvision.models as models
import torch
from vgg16_gemm import PowerMonitor, CUDALayerTimer, measure_overall_performance, format_tuple_value, print_cuda_info
import sys
import os
# Add the gemm directory to path for importing shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../gemm'))


# Add parent directories to path for imports
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../..')))

# Try to import pynvml for power monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("⚠ WARNING: pynvml not available. Power monitoring will be disabled.")
    print("  Install with: pip install nvidia-ml-py3")


# Import shared classes from gemm script
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../gemm')))


def main():
    """Main profiling function"""
    print("=" * 80)
    print("VGG16 DIRECT PERFORMANCE + POWER PROFILING - FULL VERSION")
    print("=" * 80)

    # Configuration
    MODEL_NAME = "VGG16"
    ALGORITHM = "direct"
    BATCH_SIZE = 1
    WARMUP_ITERS = 10
    MEASURE_ITERS = 20

    # FULL VERSION: 100 input sizes from 224 to 512
    INPUT_SIZES = [224 + int(i * (512 - 224) / 99) for i in range(100)]

    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Algorithm: {ALGORITHM}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Warmup iterations: {WARMUP_ITERS}")
    print(f"  Measurement iterations: {MEASURE_ITERS}")
    print(f"  Number of input sizes: {len(INPUT_SIZES)}")
    print(f"  Input size range: {INPUT_SIZES[0]} to {INPUT_SIZES[-1]}")
    print(f"  Sample sizes: {INPUT_SIZES[:5]} ... {INPUT_SIZES[-5:]}")

    # Initialize power monitoring
    print("\nInitializing power monitoring...")
    power_monitor = PowerMonitor()
    if power_monitor.enabled:
        power_limit = power_monitor.get_power_limit()
        print(f"  ✓ Power limit: {power_limit:.2f}W")
    else:
        print("  ⚠ Power monitoring disabled (performance metrics will still be collected)")

    # Check CUDA availability
    use_cuda = print_cuda_info()
    device = torch.device('cpu')

    # Check ai3 cuDNN support
    if not ai3.using_cudnn():
        print("\n✗ ERROR: cuDNN is not available in ai3")
        print("  This algorithm requires cuDNN support")
        print("  Please rebuild ai3 with USE_CUDNN flag")
        sys.exit(1)
    print(f"\n✓ ai3 cuDNN support confirmed")

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    try:
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.eval()
        print(f"✓ {MODEL_NAME} loaded")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)

    # Apply ai3 conversion
    print(f"\nApplying ai3 '{ALGORITHM}' algorithm conversion...")
    try:
        ai3.swap_conv2d(model, ALGORITHM)

        # Count converted layers
        conv_layers = sum(1 for _, m in model.named_modules()
                          if isinstance(m, torch.nn.Conv2d) or hasattr(m, 'algorithm'))
        ai3_layers = sum(1 for _, m in model.named_modules()
                         if hasattr(m, 'algorithm'))

        print(f"✓ Conversion completed:")
        print(f"  Total Conv2D layers: {conv_layers}")
        print(f"  AI3 converted layers: {ai3_layers}")
        print(f"  Conversion rate: {(ai3_layers/conv_layers*100):.1f}%")
    except Exception as e:
        print(f"✗ Error during ai3 conversion: {e}")
        sys.exit(1)

    print(f"\n✓ Model ready (ai3 will use GPU internally for conv operations)")

    # Initialize timer with power monitoring
    timer = CUDALayerTimer(use_cuda=use_cuda, power_monitor=power_monitor)
    timer.register_hooks(model)

    # Data collection storage
    overall_results = []
    layer_results = []

    print(f"\n{'='*80}")
    print("STARTING PERFORMANCE + POWER PROFILING")
    print(f"{'='*80}")

    # Profile each input size
    for idx, input_size in enumerate(INPUT_SIZES, 1):
        print(
            f"\n[{idx}/{len(INPUT_SIZES)}] Profiling input size: {input_size}x{input_size}")

        # Clear GPU cache
        if use_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Create input data on CPU
        try:
            input_data = torch.randn(BATCH_SIZE, 3, input_size, input_size)
            print(
                f"  ✓ Input tensor created: {tuple(input_data.shape)} on {input_data.device}")
        except Exception as e:
            print(f"  ✗ Error creating input: {e}")
            continue

        # Show GPU memory usage and current power
        if use_cuda:
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(
                f"  GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

            if power_monitor.enabled:
                current_power = power_monitor.get_power()
                if current_power:
                    print(f"  GPU Power (pre-inference): {current_power:.2f}W")

        # Measure overall performance with power
        print(f"  Measuring overall performance and power...")
        try:
            overall_stats = measure_overall_performance(
                model, input_data,
                warmup_iters=WARMUP_ITERS,
                measure_iters=MEASURE_ITERS,
                use_cuda=use_cuda,
                power_monitor=power_monitor
            )

            print(
                f"  ✓ Overall Time: {overall_stats['mean']:.2f}ms ± {overall_stats['std']:.2f}ms")
            print(
                f"    Range: [{overall_stats['min']:.2f}ms - {overall_stats['max']:.2f}ms]")
            print(f"    Median: {overall_stats['median']:.2f}ms")

            if 'power_mean_w' in overall_stats:
                print(
                    f"  ✓ Overall Power: {overall_stats['power_mean_w']:.2f}W ± {overall_stats['power_std_w']:.2f}W")
                print(
                    f"    Range: [{overall_stats['power_min_w']:.2f}W - {overall_stats['power_max_w']:.2f}W]")

            if 'energy_mean_j' in overall_stats:
                print(
                    f"  ✓ Energy per Inference: {overall_stats['energy_mean_j']:.4f}J")
                print(
                    f"    Total Energy: {overall_stats['energy_total_j']:.4f}J ({MEASURE_ITERS} runs)")

            # Store results
            device_name_for_csv = 'cuda' if use_cuda else 'cpu'
            result_dict = {
                'model': MODEL_NAME,
                'algorithm': ALGORITHM,
                'device': device_name_for_csv,
                'batch_size': BATCH_SIZE,
                'input_size': input_size,
                'mean_ms': overall_stats['mean'],
                'std_ms': overall_stats['std'],
                'min_ms': overall_stats['min'],
                'max_ms': overall_stats['max'],
                'median_ms': overall_stats['median']
            }

            # Add power metrics if available
            if 'power_mean_w' in overall_stats:
                result_dict.update({
                    'power_mean_w': overall_stats['power_mean_w'],
                    'power_std_w': overall_stats['power_std_w'],
                    'power_min_w': overall_stats['power_min_w'],
                    'power_max_w': overall_stats['power_max_w'],
                })

            if 'energy_mean_j' in overall_stats:
                result_dict.update({
                    'energy_mean_j': overall_stats['energy_mean_j'],
                    'energy_std_j': overall_stats['energy_std_j'],
                    'energy_total_j': overall_stats['energy_total_j']
                })

            overall_results.append(result_dict)
        except Exception as e:
            print(f"  ✗ Error during overall measurement: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Measure layer-wise performance
        print(f"  Measuring layer-wise performance and power...")
        timer.reset()

        try:
            # Warmup for layer timing
            with torch.inference_mode():
                for _ in range(WARMUP_ITERS):
                    _ = model(input_data)
                    if use_cuda:
                        torch.cuda.synchronize()

            timer.reset()  # Reset after warmup

            # Measurement runs
            with torch.inference_mode():
                for _ in range(MEASURE_ITERS):
                    _ = model(input_data)
                    if use_cuda:
                        torch.cuda.synchronize()

            # Collect layer statistics
            layer_stats = timer.get_statistics()
            layer_info = timer.get_layer_info()
            layer_input_sizes = timer.get_layer_input_sizes()

            # Print top 5 slowest layers with power info
            if layer_stats:
                sorted_layers = sorted(layer_stats.items(),
                                       key=lambda x: x[1]['mean'],
                                       reverse=True)[:5]
                print(f"  Top 5 slowest layers:")
                for layer_name, stats in sorted_layers:
                    percentage = (stats['mean'] / overall_stats['mean']) * 100
                    algo = layer_info.get(layer_name, {}).get(
                        'algorithm', 'unknown')
                    power_str = ""
                    if 'power_mean_w' in stats:
                        power_str = f" | {stats['power_mean_w']:.2f}W"
                    energy_str = ""
                    if 'energy_mean_j' in stats:
                        energy_str = f" | {stats['energy_mean_j']:.4f}J"
                    print(
                        f"    {layer_name}: {stats['mean']:.2f}ms ± {stats['std']:.2f}ms ({percentage:.1f}%) [{algo}]{power_str}{energy_str}")

                # Store all layer results
                for layer_name, stats in layer_stats.items():
                    info = layer_info.get(layer_name, {})
                    layer_input_size = layer_input_sizes.get(
                        layer_name, input_size)

                    layer_dict = {
                        'model': MODEL_NAME,
                        'layer': layer_name,
                        'algorithm': ALGORITHM,
                        'device': device_name_for_csv,
                        'batch_size': BATCH_SIZE,
                        'input_size': layer_input_size,
                        'in_channels': info.get('in_channels', 'N/A'),
                        'out_channels': info.get('out_channels', 'N/A'),
                        'kernel_size': format_tuple_value(info.get('kernel_size', 'N/A')),
                        'stride': format_tuple_value(info.get('stride', 'N/A')),
                        'padding': format_tuple_value(info.get('padding', 'N/A')),
                        'mean_ms': stats['mean'],
                        'std_ms': stats['std'],
                        'min_ms': stats['min'],
                        'max_ms': stats['max'],
                        'median_ms': stats['median'],
                        'percentage': (stats['mean'] / overall_stats['mean']) * 100
                    }

                    # Add power metrics if available
                    if 'power_mean_w' in stats:
                        layer_dict.update({
                            'power_mean_w': stats['power_mean_w'],
                            'power_std_w': stats['power_std_w'],
                            'power_min_w': stats['power_min_w'],
                            'power_max_w': stats['power_max_w']
                        })

                    if 'energy_mean_j' in stats:
                        layer_dict.update({
                            'energy_mean_j': stats['energy_mean_j'],
                            'energy_std_j': stats['energy_std_j'],
                            'energy_total_j': stats['energy_total_j']
                        })

                    layer_results.append(layer_dict)
            else:
                print(f"  ⚠ No layer timing data collected")

        except Exception as e:
            print(f"  ✗ Error during layer measurement: {e}")
            import traceback
            traceback.print_exc()

        # Cleanup
        del input_data
        if use_cuda:
            torch.cuda.empty_cache()

    # Remove hooks
    timer.remove_hooks()

    # Cleanup power monitor
    power_monitor.cleanup()

    # Save results to CSV
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")

    results_dir = os.getcwd()
    device_name = 'cuda' if use_cuda else 'cpu'
    overall_csv = os.path.join(
        results_dir, f"{MODEL_NAME}_{ALGORITHM}_{device_name}_overall.csv")
    layers_csv = os.path.join(
        results_dir, f"{MODEL_NAME}_{ALGORITHM}_{device_name}_layers.csv")

    # Save overall results
    try:
        with open(overall_csv, 'w', newline='') as f:
            if overall_results:
                writer = csv.DictWriter(
                    f, fieldnames=overall_results[0].keys())
                writer.writeheader()
                writer.writerows(overall_results)
        print(f"✓ Overall results saved: {overall_csv}")
    except Exception as e:
        print(f"✗ Error saving overall results: {e}")

    # Save layer results
    try:
        with open(layers_csv, 'w', newline='') as f:
            if layer_results:
                writer = csv.DictWriter(f, fieldnames=layer_results[0].keys())
                writer.writeheader()
                writer.writerows(layer_results)
        print(f"✓ Layer results saved: {layers_csv}")
    except Exception as e:
        print(f"✗ Error saving layer results: {e}")

    # Print summary
    print(f"\n{'='*80}")
    print("PROFILING COMPLETED")
    print(f"{'='*80}")
    print(f"Summary:")
    print(f"  ✓ Profiled {len(INPUT_SIZES)} input sizes")
    print(f"  ✓ {MEASURE_ITERS} iterations per size")
    print(f"  ✓ Overall results: {len(overall_results)} data points")
    print(f"  ✓ Layer results: {len(layer_results)} data points")
    print(
        f"  ✓ Power monitoring: {'ENABLED' if power_monitor.enabled else 'DISABLED'}")
    print(f"\nResults saved to:")
    print(f"  - {overall_csv}")
    print(f"  - {layers_csv}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Profiling interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
Optimized VGG16 GEMM Performance Profiling with ai3

This script provides production-quality performance profiling for VGG16
using ai3's GEMM algorithm with proper CUDA timing, statistical analysis,
and layer-wise performance breakdown.

Key improvements:
- CUDA event-based timing for accurate GPU measurements
- Proper warmup and statistical analysis
- Memory-efficient data collection
- Systematic input size sampling
- Layer-wise profiling with minimal overhead
"""

import torch
import torchvision.models as models
import ai3
import time
import os
import csv
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np

# Add parent directories to path for imports
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))


class CUDALayerTimer:
    """
    High-precision layer timer using CUDA events for accurate GPU profiling.
    Minimizes Python overhead by using GPU-based timing.
    """

    def __init__(self, use_cuda: bool = True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.layer_times = defaultdict(list)
        self.layer_info = {}
        self.layer_input_sizes = {}
        self.hooks = []
        self.current_events = {}

    def register_hooks(self, model):
        """Register forward hooks for layer timing"""
        for name, module in model.named_modules():
            # Track Conv2d and ai3 converted layers
            if (isinstance(module, torch.nn.Conv2d) or
                    hasattr(module, 'algorithm')):

                # Store layer metadata
                if isinstance(module, torch.nn.Conv2d):
                    self.layer_info[name] = {
                        'in_channels': module.in_channels,
                        'out_channels': module.out_channels,
                        'kernel_size': module.kernel_size,
                        'stride': module.stride,
                        'padding': module.padding,
                        'algorithm': 'pytorch_conv2d'
                    }
                elif hasattr(module, 'algorithm'):
                    if hasattr(module, 'weight'):
                        weight_shape = module.weight.shape
                        self.layer_info[name] = {
                            'out_channels': weight_shape[0],
                            'in_channels': weight_shape[1],
                            'kernel_size': (weight_shape[2], weight_shape[3]),
                            'algorithm': getattr(module, 'algorithm', 'ai3_unknown')
                        }
                        if hasattr(module, 'stride'):
                            self.layer_info[name]['stride'] = module.stride
                        if hasattr(module, 'padding'):
                            self.layer_info[name]['padding'] = module.padding

                # Register timing hooks
                pre_hook = module.register_forward_pre_hook(
                    self._create_pre_hook(name))
                post_hook = module.register_forward_hook(
                    self._create_post_hook(name))
                self.hooks.append(pre_hook)
                self.hooks.append(post_hook)

    def _create_pre_hook(self, name):
        """Pre-forward hook with CUDA event timing"""
        def hook(module, input):
            if self.use_cuda:
                # Use CUDA events for precise GPU timing
                start_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                self.current_events[name] = {'start': start_event}
            else:
                # CPU fallback
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                self.current_events[name] = {'start': time.time()}

            # Capture input dimensions
            if isinstance(input, tuple) and len(input) > 0:
                input_tensor = input[0]
                if hasattr(input_tensor, 'shape') and len(input_tensor.shape) >= 4:
                    batch_size, channels, height, width = input_tensor.shape
                    self.current_events[name].update({
                        'input_shape': input_tensor.shape,
                        'input_height': height,
                        'input_width': width,
                        'input_channels': channels,
                        'batch_size': batch_size
                    })
                    self.layer_input_sizes[
                        name] = height if height == width else f"{height}x{width}"
        return hook

    def _create_post_hook(self, name):
        """Post-forward hook to complete timing measurement"""
        def hook(module, input, output):
            if name not in self.current_events:
                return

            event_data = self.current_events[name]

            if self.use_cuda:
                # Record end event
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                torch.cuda.synchronize()  # Wait for completion

                # Calculate elapsed time in milliseconds
                duration_ms = event_data['start'].elapsed_time(end_event)
                event_data['duration_ms'] = duration_ms
            else:
                # CPU timing
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                duration_ms = (end_time - event_data['start']) * 1000
                event_data['duration_ms'] = duration_ms

            self.layer_times[name].append(event_data)
            del self.current_events[name]  # Clean up
        return hook

    def reset(self):
        """Reset timing data while preserving hooks"""
        self.layer_times = defaultdict(list)
        self.current_events = {}
        self.layer_input_sizes = {}

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate comprehensive statistics for each layer"""
        results = {}
        for name, times in self.layer_times.items():
            durations = [entry.get('duration_ms', 0)
                         for entry in times if 'duration_ms' in entry]
            if durations:
                results[name] = {
                    'mean': np.mean(durations),
                    'std': np.std(durations),
                    'min': np.min(durations),
                    'max': np.max(durations),
                    'median': np.median(durations),
                    'count': len(durations)
                }
        return results

    def get_layer_info(self):
        """Get layer configuration information"""
        return self.layer_info

    def get_layer_input_sizes(self):
        """Get actual input dimensions received by each layer"""
        return self.layer_input_sizes


def measure_overall_performance(
    model,
    input_data: torch.Tensor,
    warmup_iters: int = 10,
    measure_iters: int = 20,
    use_cuda: bool = True
) -> Dict[str, float]:
    """
    Measure overall model performance with proper GPU timing.

    Args:
        model: Model to benchmark
        input_data: Input tensor
        warmup_iters: Number of warmup iterations
        measure_iters: Number of measurement iterations
        use_cuda: Whether to use CUDA event timing

    Returns:
        Dictionary with timing statistics
    """
    device = input_data.device
    use_cuda_timing = use_cuda and device.type == 'cuda'

    # Warmup phase
    with torch.inference_mode():
        for _ in range(warmup_iters):
            _ = model(input_data)
            if use_cuda_timing:
                torch.cuda.synchronize()

    # Measurement phase
    times = []

    with torch.inference_mode():
        if use_cuda_timing:
            for _ in range(measure_iters):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                _ = model(input_data)
                end_event.record()

                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(
                    end_event))  # milliseconds
        else:
            for _ in range(measure_iters):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.time()
                _ = model(input_data)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.time()
                times.append((end - start) * 1000)  # Convert to ms

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times)
    }


def format_tuple_value(value):
    """Format tuple values for CSV output"""
    if isinstance(value, tuple):
        if len(value) == 1:
            return str(value[0])
        elif len(value) == 2 and value[0] == value[1]:
            return str(value[0])
        else:
            return f"{value[0]}x{value[1]}"
    return str(value)


def print_cuda_info():
    """Print CUDA device information"""
    if not torch.cuda.is_available():
        print("✗ CUDA is not available")
        return False

    print("\nCUDA Device Information:")
    print(f"  ✓ CUDA version: {torch.version.cuda}")
    print(f"  ✓ cuDNN version: {torch.backends.cudnn.version()}")
    print(f"  ✓ Number of GPUs: {torch.cuda.device_count()}")
    print(f"  ✓ Current GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  ✓ GPU Memory: {props.total_memory / 1024**3:.1f} GB")
    print(f"  ✓ Compute Capability: {props.major}.{props.minor}")

    # Initialize CUDA context
    torch.cuda.init()
    torch.cuda.empty_cache()

    # Configure cuDNN
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    print("  ✓ CUDA context initialized")
    return True


def main():
    """Main profiling function"""
    print("=" * 80)
    print("OPTIMIZED VGG16 GEMM PERFORMANCE PROFILING")
    print("=" * 80)

    # Configuration
    MODEL_NAME = "VGG16"
    ALGORITHM = "gemm"
    BATCH_SIZE = 1
    WARMUP_ITERS = 10
    MEASURE_ITERS = 20

    # Systematic input size sampling - 100 datapoints
    # Uses regular intervals from 224 to 512 (inclusive)
    # Much better than random sampling: predictable, reproducible, good coverage
    # Step size: (512 - 224) / 99 ≈ 2.91 → ~3 pixels between each size
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

    # Check CUDA availability
    use_cuda = print_cuda_info()
    # Note: ai3 handles GPU internally, model stays on CPU
    device = torch.device('cpu')

    # Check ai3 cuDNN support
    if ALGORITHM in ['gemm', 'implicit_gemm', 'implicit_precomp_gemm', 'winograd', 'winograd_nonfused']:
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
    print(f"\nApplying ai3 {ALGORITHM} algorithm conversion...")
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

    # Note: Keep model on CPU - ai3 handles GPU transfers internally
    # This is required because ai3 layers expect CPU input/output interface
    # while using GPU internally for computation
    print(f"\n✓ Model ready (ai3 will use GPU internally for conv operations)")

    # Initialize timer
    timer = CUDALayerTimer(use_cuda=use_cuda)
    timer.register_hooks(model)

    # Data collection storage
    overall_results = []
    layer_results = []

    print(f"\n{'='*80}")
    print("STARTING PERFORMANCE PROFILING")
    print(f"{'='*80}")

    # Profile each input size
    for idx, input_size in enumerate(INPUT_SIZES, 1):
        print(
            f"\n[{idx}/{len(INPUT_SIZES)}] Profiling input size: {input_size}x{input_size}")

        # Clear GPU cache
        if use_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Create input data on CPU (ai3 may handle device transfer internally)
        try:
            input_data = torch.randn(
                BATCH_SIZE, 3, input_size, input_size)  # CPU tensor
            print(
                f"  ✓ Input tensor created: {tuple(input_data.shape)} on {input_data.device}")
        except Exception as e:
            print(f"  ✗ Error creating input: {e}")
            continue

        # Show GPU memory usage
        if use_cuda:
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(
                f"  GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

        # Measure overall performance
        print(f"  Measuring overall performance...")
        try:
            overall_stats = measure_overall_performance(
                model, input_data,
                warmup_iters=WARMUP_ITERS,
                measure_iters=MEASURE_ITERS,
                use_cuda=use_cuda
            )

            print(
                f"  ✓ Overall: {overall_stats['mean']:.2f}ms ± {overall_stats['std']:.2f}ms")
            print(
                f"    Range: [{overall_stats['min']:.2f}ms - {overall_stats['max']:.2f}ms]")
            print(f"    Median: {overall_stats['median']:.2f}ms")

            # Store results
            device_name_for_csv = 'cuda' if use_cuda else 'cpu'
            overall_results.append({
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
            })
        except Exception as e:
            print(f"  ✗ Error during overall measurement: {e}")
            continue

        # Measure layer-wise performance
        print(f"  Measuring layer-wise performance...")
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

            # Print top 5 slowest layers
            if layer_stats:
                sorted_layers = sorted(layer_stats.items(),
                                       key=lambda x: x[1]['mean'],
                                       reverse=True)[:5]
                print(f"  Top 5 slowest layers:")
                for layer_name, stats in sorted_layers:
                    percentage = (stats['mean'] / overall_stats['mean']) * 100
                    algo = layer_info.get(layer_name, {}).get(
                        'algorithm', 'unknown')
                    print(
                        f"    {layer_name}: {stats['mean']:.2f}ms ± {stats['std']:.2f}ms ({percentage:.1f}%) [{algo}]")

                # Store all layer results
                for layer_name, stats in layer_stats.items():
                    info = layer_info.get(layer_name, {})
                    layer_input_size = layer_input_sizes.get(
                        layer_name, input_size)

                    layer_results.append({
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
                    })
            else:
                print(f"  ⚠ No layer timing data collected")

        except Exception as e:
            print(f"  ✗ Error during layer measurement: {e}")

        # Cleanup
        del input_data
        if use_cuda:
            torch.cuda.empty_cache()

    # Remove hooks
    timer.remove_hooks()

    # Save results to CSV
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")

    results_dir = os.getcwd()
    # Use 'cuda' in filename since ai3 uses GPU internally even though model interface is CPU
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

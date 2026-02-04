#!/usr/bin/env python3
"""
Unified GPU Performance Profiler for ai3

This module provides a consolidated profiling system that works with any
model + algorithm combination. It eliminates code duplication across
individual model/algorithm scripts.

Features:
- CUDA event-based precise timing
- Power monitoring via NVML (pynvml)
- Layer-wise breakdown with hooks
- Trapezoidal rule for accurate power integration
- CSV output for analysis

Usage:
    from profiler import UnifiedProfiler
    
    profiler = UnifiedProfiler(
        model_name='vgg16',
        algorithm='gemm',
        num_sizes=100
    )
    profiler.run(output_dir='./results')
"""

import torch
import torchvision.models as models
import ai3
import time
import os
import csv
import sys
import threading
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np

# Add parent directory for ai3 imports
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

# Try to import pynvml for power monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("⚠ WARNING: pynvml not available. Power monitoring will be disabled.")
    print("  Install with: pip install nvidia-ml-py3")


# ============================================================================
# MODEL REGISTRY
# ============================================================================
MODEL_REGISTRY: Dict[str, Callable] = {
    'vgg16': lambda: models.vgg16(weights=models.VGG16_Weights.DEFAULT),
    'densenet121': lambda: models.densenet121(weights=models.DenseNet121_Weights.DEFAULT),
    'resnet152': lambda: models.resnet152(weights=models.ResNet152_Weights.DEFAULT),
    'googlenet': lambda: models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT),
    'alexnet': lambda: models.alexnet(weights=models.AlexNet_Weights.DEFAULT),
    'resnet50': lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
}

# ============================================================================
# ALGORITHM REGISTRY
# ============================================================================
ALGORITHMS = [
    'direct',
    'gemm',
    'implicit gemm',
    'implicit precomp gemm',
    'fft',
    'fft tiling',
    'winograd',
    'winograd nonfused',
    'guess',
]

# Algorithms that only work with specific models (e.g., 3x3 kernels only)
ALGORITHM_COMPATIBILITY = {
    'winograd': ['vgg16'],           # Winograd needs 3x3 kernels
    'winograd nonfused': ['vgg16'],  # Same restriction
}


# ============================================================================
# POWER MONITOR CLASS
# ============================================================================
class PowerMonitor:
    """
    GPU power monitoring using NVIDIA Management Library (NVML).

    Provides:
    - Real-time power readings (Watts)
    - Power limit information
    - Threaded continuous sampling for trapezoidal integration
    """

    def __init__(self):
        self.enabled = False
        self.handle = None
        self._sampling = False
        self._samples: List[Tuple[float, float]] = []  # (timestamp, power)
        self._sample_thread = None

        if not PYNVML_AVAILABLE:
            return

        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.enabled = True

            # Test reading power
            power = self.get_power()
            if power is not None:
                print(
                    f"✓ Power monitoring initialized (current: {power:.2f}W)")
            else:
                self.enabled = False
                print("⚠ Power monitoring not available on this GPU")
        except Exception as e:
            print(f"⚠ Could not initialize power monitoring: {e}")
            self.enabled = False

    def get_power(self) -> Optional[float]:
        """Get current power draw in Watts"""
        if not self.enabled:
            return None
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            return power_mw / 1000.0  # Convert milliwatts to watts
        except Exception:
            return None

    def get_power_limit(self) -> Optional[float]:
        """Get power limit in Watts"""
        if not self.enabled:
            return None
        try:
            limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(self.handle)
            return limit_mw / 1000.0
        except Exception:
            return None

    def start_sampling(self, interval_ms: float = 1.0):
        """Start continuous power sampling in background thread"""
        if not self.enabled:
            return

        self._samples = []
        self._sampling = True

        def sample_loop():
            while self._sampling:
                power = self.get_power()
                if power is not None:
                    self._samples.append((time.perf_counter(), power))
                time.sleep(interval_ms / 1000.0)

        self._sample_thread = threading.Thread(target=sample_loop, daemon=True)
        self._sample_thread.start()

    def stop_sampling(self) -> List[Tuple[float, float]]:
        """Stop sampling and return collected samples"""
        self._sampling = False
        if self._sample_thread:
            self._sample_thread.join(timeout=0.1)
        return self._samples.copy()

    def compute_energy_trapezoidal(self, samples: List[Tuple[float, float]]) -> Dict[str, float]:
        """
        Compute energy using trapezoidal rule (professor's method).

        Formula: E = Σ [(P_i + P_{i+1}) / 2] × Δt

        Args:
            samples: List of (timestamp, power) tuples

        Returns:
            Dictionary with energy (J), avg_power (W), duration (s)
        """
        if len(samples) < 2:
            return {'energy_j': 0, 'avg_power_w': 0, 'duration_s': 0}

        total_energy = 0.0
        for i in range(len(samples) - 1):
            t1, p1 = samples[i]
            t2, p2 = samples[i + 1]
            dt = t2 - t1
            avg_power = (p1 + p2) / 2.0
            total_energy += avg_power * dt

        duration = samples[-1][0] - samples[0][0]
        powers = [p for _, p in samples]

        return {
            'energy_j': total_energy,
            'avg_power_w': np.mean(powers),
            'power_std_w': np.std(powers),
            'power_min_w': np.min(powers),
            'power_max_w': np.max(powers),
            'duration_s': duration,
            'num_samples': len(samples)
        }

    def cleanup(self):
        """Cleanup NVML"""
        self._sampling = False
        if self.enabled:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


# ============================================================================
# CUDA LAYER TIMER CLASS
# ============================================================================
class CUDALayerTimer:
    """
    High-precision layer timer using CUDA events.

    Uses PyTorch forward hooks to capture timing for each layer individually.
    """

    def __init__(self, use_cuda: bool = True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.layer_times: Dict[str, List[float]] = defaultdict(list)
        self.layer_info: Dict[str, Dict] = {}
        self.layer_input_sizes: Dict[str, int] = {}
        self.hooks: List = []
        self.current_events: Dict = {}

    def register_hooks(self, model):
        """Register forward hooks for layer timing"""
        for name, module in model.named_modules():
            # Track Conv2d and ai3 converted layers
            if isinstance(module, torch.nn.Conv2d) or hasattr(module, 'algorithm'):
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
                            'kernel_size': (weight_shape[2], weight_shape[3]) if len(weight_shape) > 3 else 'N/A',
                            'algorithm': getattr(module, 'algorithm', 'ai3_unknown')
                        }
                        if hasattr(module, 'stride'):
                            self.layer_info[name]['stride'] = module.stride
                        if hasattr(module, 'padding'):
                            self.layer_info[name]['padding'] = module.padding

                # Register hooks
                pre_hook = module.register_forward_pre_hook(
                    self._create_pre_hook(name))
                post_hook = module.register_forward_hook(
                    self._create_post_hook(name))
                self.hooks.append(pre_hook)
                self.hooks.append(post_hook)

    def _create_pre_hook(self, name):
        """Pre-forward hook to start timing"""
        def hook(module, input):
            if self.use_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                self.current_events[name] = {'start': start_event}
            else:
                self.current_events[name] = {'start': time.time()}

            # Capture input dimensions
            if isinstance(input, tuple) and len(input) > 0:
                input_tensor = input[0]
                if hasattr(input_tensor, 'shape') and len(input_tensor.shape) >= 4:
                    _, _, h, w = input_tensor.shape
                    self.layer_input_sizes[name] = h if h == w else f"{h}x{w}"
        return hook

    def _create_post_hook(self, name):
        """Post-forward hook to complete timing"""
        def hook(module, input, output):
            if name not in self.current_events:
                return

            if self.use_cuda:
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                torch.cuda.synchronize()
                duration_ms = self.current_events[name]['start'].elapsed_time(
                    end_event)
            else:
                duration_ms = (
                    time.time() - self.current_events[name]['start']) * 1000

            self.layer_times[name].append(duration_ms)
            del self.current_events[name]
        return hook

    def reset(self):
        """Reset timing data"""
        self.layer_times = defaultdict(list)
        self.current_events = {}

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for each layer"""
        results = {}
        for name, times in self.layer_times.items():
            if times:
                results[name] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'median': np.median(times),
                    'count': len(times)
                }
        return results

    def get_layer_info(self):
        return self.layer_info

    def get_layer_input_sizes(self):
        return self.layer_input_sizes


# ============================================================================
# UNIFIED PROFILER CLASS
# ============================================================================
class UnifiedProfiler:
    """
    Main profiler class that handles any model + algorithm combination.

    Args:
        model_name: Name of model from MODEL_REGISTRY
        algorithm: Convolution algorithm to use
        batch_size: Batch size for inputs
        num_sizes: Number of input sizes to test
        warmup_iters: Warmup iterations before measuring
        measure_iters: Measurement iterations
        input_size_range: Tuple of (min_size, max_size)
    """

    def __init__(
        self,
        model_name: str,
        algorithm: str,
        batch_size: int = 1,
        num_sizes: int = 100,
        warmup_iters: int = 10,
        measure_iters: int = 20,
        input_size_range: Tuple[int, int] = (224, 512)
    ):
        self.model_name = model_name.lower()
        self.algorithm = algorithm.lower()
        self.batch_size = batch_size
        self.warmup_iters = warmup_iters
        self.measure_iters = measure_iters

        # Generate input sizes
        min_size, max_size = input_size_range
        if num_sizes == 1:
            self.input_sizes = [min_size]
        else:
            self.input_sizes = [
                min_size + int(i * (max_size - min_size) / (num_sizes - 1))
                for i in range(num_sizes)
            ]

        self.power_monitor = PowerMonitor()
        self.timer: Optional[CUDALayerTimer] = None
        self.model = None
        self.use_cuda = torch.cuda.is_available()

    def _check_compatibility(self):
        """Check if algorithm is compatible with model"""
        if self.algorithm in ALGORITHM_COMPATIBILITY:
            compatible_models = ALGORITHM_COMPATIBILITY[self.algorithm]
            if self.model_name not in compatible_models:
                raise ValueError(
                    f"Algorithm '{self.algorithm}' is only compatible with: "
                    f"{compatible_models}, not '{self.model_name}'"
                )

    def _load_model(self):
        """Load model from registry and apply ai3 conversion"""
        if self.model_name not in MODEL_REGISTRY:
            available = list(MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unknown model: {self.model_name}. Available: {available}")

        print(f"\nLoading {self.model_name}...")
        self.model = MODEL_REGISTRY[self.model_name]()
        self.model.eval()
        print(f"✓ {self.model_name} loaded")

        # Check cuDNN availability for algorithms that need it
        cudnn_algorithms = ['gemm', 'implicit gemm', 'implicit precomp gemm',
                            'fft', 'fft tiling', 'winograd', 'winograd nonfused']
        if self.algorithm in cudnn_algorithms:
            if not ai3.using_cudnn():
                raise RuntimeError(
                    f"Algorithm '{self.algorithm}' requires cuDNN, but cuDNN is not available. "
                    "Please rebuild ai3 with USE_CUDNN flag."
                )

        # Apply ai3 conversion
        print(f"Applying ai3 '{self.algorithm}' algorithm conversion...")
        ai3.swap_conv2d(self.model, self.algorithm)

        # Count converted layers
        conv_layers = sum(1 for _, m in self.model.named_modules()
                          if isinstance(m, torch.nn.Conv2d) or hasattr(m, 'algorithm'))
        ai3_layers = sum(1 for _, m in self.model.named_modules()
                         if hasattr(m, 'algorithm'))

        print(f"✓ Conversion completed:")
        print(f"  Total Conv2D layers: {conv_layers}")
        print(f"  AI3 converted layers: {ai3_layers}")
        if conv_layers > 0:
            print(f"  Conversion rate: {(ai3_layers/conv_layers*100):.1f}%")

    def _measure_overall(self, input_data: torch.Tensor) -> Dict[str, float]:
        """Measure overall model performance with power monitoring"""

        # Warmup
        with torch.inference_mode():
            for _ in range(self.warmup_iters):
                _ = self.model(input_data)
                if self.use_cuda:
                    torch.cuda.synchronize()

        # Measurement with power sampling
        times = []
        all_power_samples = []

        with torch.inference_mode():
            for _ in range(self.measure_iters):
                # Start power sampling
                self.power_monitor.start_sampling(interval_ms=0.5)

                if self.use_cuda:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    start_event.record()
                    _ = self.model(input_data)
                    end_event.record()
                    torch.cuda.synchronize()

                    duration_ms = start_event.elapsed_time(end_event)
                else:
                    start = time.time()
                    _ = self.model(input_data)
                    duration_ms = (time.time() - start) * 1000

                # Stop power sampling
                samples = self.power_monitor.stop_sampling()

                times.append(duration_ms)
                if samples:
                    all_power_samples.extend(samples)

        # Calculate statistics
        stats = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times)
        }

        # Add power statistics using trapezoidal rule
        if all_power_samples:
            power_stats = self.power_monitor.compute_energy_trapezoidal(
                all_power_samples)
            stats.update({
                'power_mean_w': power_stats['avg_power_w'],
                'power_std_w': power_stats['power_std_w'],
                'power_min_w': power_stats['power_min_w'],
                'power_max_w': power_stats['power_max_w'],
                'energy_total_j': power_stats['energy_j'],
                'energy_per_inference_j': power_stats['energy_j'] / self.measure_iters,
                'power_samples': power_stats['num_samples']
            })

        return stats

    def _measure_layers(self, input_data: torch.Tensor) -> List[Dict]:
        """Measure layer-wise performance"""
        self.timer.reset()

        # Warmup
        with torch.inference_mode():
            for _ in range(self.warmup_iters):
                _ = self.model(input_data)
                if self.use_cuda:
                    torch.cuda.synchronize()

        self.timer.reset()

        # Measurement
        with torch.inference_mode():
            for _ in range(self.measure_iters):
                _ = self.model(input_data)
                if self.use_cuda:
                    torch.cuda.synchronize()

        return self.timer.get_statistics()

    def run(self, output_dir: str = "./results") -> Dict:
        """
        Run the full profiling pipeline.

        Args:
            output_dir: Directory to save results

        Returns:
            Dictionary with 'overall' and 'layers' results
        """
        print("=" * 70)
        print(
            f"UNIFIED PROFILER: {self.model_name.upper()} + {self.algorithm.upper()}")
        print("=" * 70)

        print(f"\nConfiguration:")
        print(f"  Model: {self.model_name}")
        print(f"  Algorithm: {self.algorithm}")
        print(f"  Batch size: {self.batch_size}")
        print(
            f"  Input sizes: {len(self.input_sizes)} ({self.input_sizes[0]} to {self.input_sizes[-1]})")
        print(f"  Warmup iterations: {self.warmup_iters}")
        print(f"  Measurement iterations: {self.measure_iters}")

        # Check compatibility and load model
        self._check_compatibility()
        self._load_model()

        # Print CUDA info
        if self.use_cuda:
            print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  cuDNN version: {torch.backends.cudnn.version()}")
        else:
            print("\n⚠ CUDA not available, running on CPU")

        # Setup timer with hooks
        self.timer = CUDALayerTimer(use_cuda=self.use_cuda)
        self.timer.register_hooks(self.model)

        # Data collection
        overall_results = []
        layer_results = []

        print(f"\n{'='*70}")
        print("STARTING PROFILING")
        print(f"{'='*70}")

        for idx, input_size in enumerate(self.input_sizes, 1):
            print(
                f"\n[{idx}/{len(self.input_sizes)}] Input size: {input_size}x{input_size}")

            if self.use_cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Create input
            try:
                input_data = torch.randn(
                    self.batch_size, 3, input_size, input_size)
                print(f"  ✓ Input tensor: {tuple(input_data.shape)}")
            except Exception as e:
                print(f"  ✗ Error creating input: {e}")
                continue

            # Measure overall
            try:
                overall_stats = self._measure_overall(input_data)
                print(
                    f"  ✓ Time: {overall_stats['mean']:.2f}ms ± {overall_stats['std']:.2f}ms")

                if 'power_mean_w' in overall_stats:
                    print(
                        f"  ✓ Power: {overall_stats['power_mean_w']:.2f}W (trapezoidal, {overall_stats.get('power_samples', 'N/A')} samples)")
                    print(
                        f"  ✓ Energy: {overall_stats.get('energy_per_inference_j', 0):.4f}J per inference")

                overall_results.append({
                    'model': self.model_name,
                    'algorithm': self.algorithm,
                    'batch_size': self.batch_size,
                    'input_size': input_size,
                    **overall_stats
                })
            except Exception as e:
                print(f"  ✗ Error during measurement: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Measure layers
            try:
                layer_stats = self._measure_layers(input_data)
                layer_info = self.timer.get_layer_info()
                layer_input_sizes = self.timer.get_layer_input_sizes()

                # Top 3 slowest layers
                if layer_stats:
                    sorted_layers = sorted(layer_stats.items(),
                                           key=lambda x: x[1]['mean'],
                                           reverse=True)[:3]
                    print(f"  Top 3 slowest layers:")
                    for layer_name, stats in sorted_layers:
                        pct = (stats['mean'] / overall_stats['mean']) * 100
                        print(
                            f"    {layer_name}: {stats['mean']:.2f}ms ({pct:.1f}%)")

                    # Store layer results
                    for layer_name, stats in layer_stats.items():
                        info = layer_info.get(layer_name, {})
                        layer_results.append({
                            'model': self.model_name,
                            'algorithm': self.algorithm,
                            'input_size': input_size,
                            'layer': layer_name,
                            'in_channels': info.get('in_channels', 'N/A'),
                            'out_channels': info.get('out_channels', 'N/A'),
                            'kernel_size': str(info.get('kernel_size', 'N/A')),
                            'mean_ms': stats['mean'],
                            'std_ms': stats['std'],
                            'percentage': (stats['mean'] / overall_stats['mean']) * 100
                        })
            except Exception as e:
                print(f"  ✗ Error during layer measurement: {e}")

            del input_data
            if self.use_cuda:
                torch.cuda.empty_cache()

        # Cleanup
        self.timer.remove_hooks()
        self.power_monitor.cleanup()

        # Save results
        self._save_results(overall_results, layer_results, output_dir)

        print(f"\n{'='*70}")
        print("PROFILING COMPLETED")
        print(f"{'='*70}")
        print(f"  ✓ {len(overall_results)} input sizes profiled")
        print(f"  ✓ {len(layer_results)} layer measurements")
        print(f"  ✓ Results saved to: {output_dir}")

        return {'overall': overall_results, 'layers': layer_results}

    def _save_results(self, overall: List[Dict], layers: List[Dict], output_dir: str):
        """Save results to CSV files"""
        os.makedirs(output_dir, exist_ok=True)

        algo_safe = self.algorithm.replace(' ', '_')
        base_name = f"{self.model_name}_{algo_safe}"

        # Save overall results
        if overall:
            overall_path = os.path.join(output_dir, f"{base_name}_overall.csv")
            with open(overall_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=overall[0].keys())
                writer.writeheader()
                writer.writerows(overall)
            print(f"\n✓ Overall results: {overall_path}")

        # Save layer results
        if layers:
            layers_path = os.path.join(output_dir, f"{base_name}_layers.csv")
            with open(layers_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=layers[0].keys())
                writer.writeheader()
                writer.writerows(layers)
            print(f"✓ Layer results: {layers_path}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
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

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    return True


def list_available_models():
    """List all available models"""
    return list(MODEL_REGISTRY.keys())


def list_available_algorithms():
    """List all available algorithms"""
    return ALGORITHMS.copy()


def check_compatibility(model_name: str, algorithm: str) -> bool:
    """Check if model and algorithm are compatible"""
    if algorithm in ALGORITHM_COMPATIBILITY:
        return model_name in ALGORITHM_COMPATIBILITY[algorithm]
    return True

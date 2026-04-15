#!/usr/bin/env python3
"""
Unified GPU Performance Profiler for ai3 -- v2 with per-layer power/energy.

Extends the original profiler with continuous NVML sampling that is sliced
per-layer using timestamp markers recorded in forward hooks.  This avoids
the overhead of starting/stopping a sampling thread for every layer.

Features:
- CUDA event-based precise timing
- Per-layer AND overall power monitoring via NVML (pynvml)
- Continuous background power sampling with per-layer timestamp slicing
- Trapezoidal rule for accurate energy integration when enough NVML samples exist
- Hybrid rule: 1 sample uses E ~= P * wall_duration; 0 samples fall back to
  proportional allocation from overall energy_per_inference_j
- Faster default NVML polling during layer measurement (configurable)
- Layer-level CSV: power_mean_w, energy_per_layer_j, power_samples, energy_attribution
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

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("WARNING: pynvml not available. Power monitoring will be disabled.")
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

ALGORITHM_COMPATIBILITY = {
    'winograd': ['vgg16'],
    'winograd nonfused': ['vgg16'],
}


# ============================================================================
# POWER MONITOR CLASS
# ============================================================================
class PowerMonitor:
    """
    GPU power monitoring using NVIDIA Management Library (NVML).

    Supports two usage patterns:
      1. start_sampling / stop_sampling  -- one inference at a time (overall)
      2. start_continuous_sampling / stop_continuous_sampling + slice_samples
         -- long-running background thread for per-layer energy attribution
    """

    def __init__(self):
        self.enabled = False
        self.handle = None
        self._sampling = False
        self._samples: List[Tuple[float, float]] = []
        self._continuous_samples: List[Tuple[float, float]] = []
        self._sample_thread: Optional[threading.Thread] = None

        if not PYNVML_AVAILABLE:
            return

        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.enabled = True

            power = self.get_power()
            if power is not None:
                print(f"  Power monitoring initialized (current: {power:.2f}W)")
            else:
                self.enabled = False
                print("  Power monitoring not available on this GPU")
        except Exception as e:
            print(f"  Could not initialize power monitoring: {e}")
            self.enabled = False

    def get_power(self) -> Optional[float]:
        """Get current power draw in Watts."""
        if not self.enabled:
            return None
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            return power_mw / 1000.0
        except Exception:
            return None

    def get_power_limit(self) -> Optional[float]:
        """Get power limit in Watts."""
        if not self.enabled:
            return None
        try:
            limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(self.handle)
            return limit_mw / 1000.0
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Pattern 1: per-inference sampling (used by _measure_overall)
    # ------------------------------------------------------------------
    def start_sampling(self, interval_ms: float = 1.0):
        """Start power sampling in a background thread (single inference)."""
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
        """Stop per-inference sampling and return samples."""
        self._sampling = False
        if self._sample_thread:
            self._sample_thread.join(timeout=0.1)
        return self._samples.copy()

    # ------------------------------------------------------------------
    # Pattern 2: continuous sampling (used by _measure_layers)
    # ------------------------------------------------------------------
    def start_continuous_sampling(self, interval_ms: float = 0.02):
        """
        Start a long-running background sampling thread.

        Runs across many iterations so per-layer timestamps can be matched
        to power readings after the fact via slice_samples().

        Default interval_ms=0.02 (~20 us between polls, requested; OS may
        quantize ``time.sleep`` to coarser resolution).
        """
        if not self.enabled:
            return
        self._continuous_samples = []
        self._sampling = True
        sleep_sec = max(interval_ms / 1000.0, 1e-6)

        def sample_loop():
            while self._sampling:
                power = self.get_power()
                if power is not None:
                    self._continuous_samples.append(
                        (time.perf_counter(), power))
                time.sleep(sleep_sec)

        self._sample_thread = threading.Thread(target=sample_loop, daemon=True)
        self._sample_thread.start()

    def stop_continuous_sampling(self) -> List[Tuple[float, float]]:
        """Stop continuous sampling and return all collected samples."""
        self._sampling = False
        if self._sample_thread:
            self._sample_thread.join(timeout=0.5)
        return self._continuous_samples.copy()

    def slice_samples(
        self,
        all_samples: List[Tuple[float, float]],
        t_start: float,
        t_end: float,
        eps: float = 1e-7,
    ) -> List[Tuple[float, float]]:
        """Return samples whose timestamp falls in [t_start, t_end] (with tiny eps)."""
        lo = t_start - eps
        hi = t_end + eps
        return [(t, p) for t, p in all_samples if lo <= t <= hi]

    def compute_layer_energy_hybrid(
        self,
        samples: List[Tuple[float, float]],
        ts_list: List[Tuple[float, float]],
    ) -> Tuple[Dict[str, float], str]:
        """
        Per-layer energy from NVML samples + hook wall-clock spans.

        - len(samples) >= 2: trapezoidal integration (nvml_trapezoid)
        - len(samples) == 1: E = P * wall_duration (nvml_single_sample)
        - len(samples) == 0: zeros, attribution 'none' (caller may fallback)

        wall_duration is sum of (t_end - t_start) across measurement iterations.
        """
        wall_duration_s = sum(
            max(0.0, te - ts) for ts, te in ts_list
        )
        n = len(samples)
        if n >= 2:
            out = self.compute_energy_trapezoidal(samples)
            return out, 'nvml_trapezoid'
        if n == 1:
            p = samples[0][1]
            energy_j = p * wall_duration_s if wall_duration_s > 0 else 0.0
            return {
                'energy_j': energy_j,
                'avg_power_w': p,
                'power_std_w': 0.0,
                'power_min_w': p,
                'power_max_w': p,
                'duration_s': wall_duration_s,
                'num_samples': 1,
            }, 'nvml_single_sample'
        return {
            'energy_j': 0.0,
            'avg_power_w': 0.0,
            'power_std_w': 0.0,
            'power_min_w': 0.0,
            'power_max_w': 0.0,
            'duration_s': wall_duration_s,
            'num_samples': 0,
        }, 'none'

    # ------------------------------------------------------------------
    # Energy computation
    # ------------------------------------------------------------------
    def compute_energy_trapezoidal(
        self, samples: List[Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        Compute energy using trapezoidal rule.

        Formula: E = sum [(P_i + P_{i+1}) / 2] * dt
        """
        if len(samples) < 2:
            return {
                'energy_j': 0, 'avg_power_w': 0, 'power_std_w': 0,
                'power_min_w': 0, 'power_max_w': 0, 'duration_s': 0,
                'num_samples': len(samples),
            }

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
            'num_samples': len(samples),
        }

    def cleanup(self):
        """Cleanup NVML."""
        self._sampling = False
        if self.enabled:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


# ============================================================================
# CUDA LAYER TIMER CLASS  (v2 -- with per-layer power timestamps)
# ============================================================================
class CUDALayerTimer:
    """
    High-precision layer timer using CUDA events.

    v2 additions:
    - Records wall-clock timestamp pairs (start, end) per layer per iteration
      so that a continuous power sample stream can be sliced per-layer.
    """

    def __init__(self, use_cuda: bool = True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.layer_times: Dict[str, List[float]] = defaultdict(list)
        self.layer_info: Dict[str, Dict] = {}
        self.layer_input_sizes: Dict[str, int] = {}
        self.hooks: List = []
        self.current_events: Dict = {}
        # v2: wall-clock timestamp windows per layer per iteration
        self.layer_timestamps: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    def register_hooks(self, model):
        """Register forward hooks for layer timing and timestamp capture."""
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or hasattr(module, 'algorithm'):
                if isinstance(module, torch.nn.Conv2d):
                    self.layer_info[name] = {
                        'in_channels': module.in_channels,
                        'out_channels': module.out_channels,
                        'kernel_size': module.kernel_size,
                        'stride': module.stride,
                        'padding': module.padding,
                        'algorithm': 'pytorch_conv2d',
                    }
                elif hasattr(module, 'algorithm'):
                    if hasattr(module, 'weight'):
                        weight_shape = module.weight.shape
                        self.layer_info[name] = {
                            'out_channels': weight_shape[0],
                            'in_channels': weight_shape[1],
                            'kernel_size': (
                                (weight_shape[2], weight_shape[3])
                                if len(weight_shape) > 3 else 'N/A'
                            ),
                            'algorithm': getattr(module, 'algorithm', 'ai3_unknown'),
                        }
                        if hasattr(module, 'stride'):
                            self.layer_info[name]['stride'] = module.stride
                        if hasattr(module, 'padding'):
                            self.layer_info[name]['padding'] = module.padding

                pre_hook = module.register_forward_pre_hook(
                    self._create_pre_hook(name))
                post_hook = module.register_forward_hook(
                    self._create_post_hook(name))
                self.hooks.append(pre_hook)
                self.hooks.append(post_hook)

    def _create_pre_hook(self, name):
        def hook(module, input):
            wall_start = time.perf_counter()
            if self.use_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                self.current_events[name] = {
                    'start': start_event,
                    'wall_start': wall_start,
                }
            else:
                self.current_events[name] = {
                    'start': wall_start,
                    'wall_start': wall_start,
                }

            if isinstance(input, tuple) and len(input) > 0:
                input_tensor = input[0]
                if hasattr(input_tensor, 'shape') and len(input_tensor.shape) >= 4:
                    _, _, h, w = input_tensor.shape
                    self.layer_input_sizes[name] = h if h == w else f"{h}x{w}"
        return hook

    def _create_post_hook(self, name):
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
                    time.perf_counter() - self.current_events[name]['start']
                ) * 1000

            wall_end = time.perf_counter()

            self.layer_times[name].append(duration_ms)
            self.layer_timestamps[name].append(
                (self.current_events[name]['wall_start'], wall_end)
            )
            del self.current_events[name]
        return hook

    def reset(self):
        """Reset timing and timestamp data."""
        self.layer_times = defaultdict(list)
        self.layer_timestamps = defaultdict(list)
        self.current_events = {}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate timing statistics for each layer."""
        results = {}
        for name, times in self.layer_times.items():
            if times:
                results[name] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'median': np.median(times),
                    'count': len(times),
                }
        return results

    def get_power_statistics(
        self,
        power_monitor: 'PowerMonitor',
        all_samples: List[Tuple[float, float]],
        timing_stats: Dict[str, Dict[str, float]],
        energy_per_inference_j: Optional[float],
        measure_iters: int,
    ) -> Dict[str, Dict]:
        """
        Per-layer power/energy: NVML slice + hybrid rule + proportional fallback.

        Fallback (proportional_overall): energy_per_layer_j =
            energy_per_inference_j * (layer_mean_ms / sum_conv_mean_ms)
        when NVML yields no usable energy for that layer.
        """
        total_ms = sum(v['mean'] for v in timing_stats.values())
        budget = float(energy_per_inference_j or 0.0)

        results: Dict[str, Dict] = {}
        for name, ts_list in self.layer_timestamps.items():
            layer_samples: List[Tuple[float, float]] = []
            for t_start, t_end in ts_list:
                layer_samples.extend(
                    power_monitor.slice_samples(all_samples, t_start, t_end)
                )

            stats, attr = power_monitor.compute_layer_energy_hybrid(
                layer_samples, ts_list)
            nit = len(ts_list) if ts_list else measure_iters
            nit = max(nit, 1)

            epl = stats['energy_j'] / nit
            pmw = float(stats['avg_power_w'])
            pstd = float(stats['power_std_w'])
            ps = int(stats['num_samples'])
            etot = float(stats['energy_j'])

            need_fallback = (
                epl <= 0.0
                and budget > 0.0
                and total_ms > 0.0
                and name in timing_stats
            )
            if need_fallback:
                mean_ms = float(timing_stats[name]['mean'])
                epl = budget * (mean_ms / total_ms)
                dt_s = mean_ms / 1000.0
                pmw = epl / dt_s if dt_s > 0 else 0.0
                pstd = 0.0
                etot = epl * nit
                ps = 0
                attr = 'proportional_overall'

            results[name] = {
                'power_mean_w': pmw,
                'power_std_w': pstd,
                'energy_total_j': etot,
                'energy_per_layer_j': epl,
                'power_samples': ps,
                'energy_attribution': attr,
            }
        return results

    def get_layer_info(self):
        return self.layer_info

    def get_layer_input_sizes(self):
        return self.layer_input_sizes


# ============================================================================
# UNIFIED PROFILER CLASS (v2)
# ============================================================================
class UnifiedProfiler:
    """
    Main profiler class that handles any model + algorithm combination.

    v2: layer measurement now also captures per-layer power/energy via
    continuous NVML sampling + timestamp slicing.
    """

    def __init__(
        self,
        model_name: str,
        algorithm: str,
        batch_size: int = 1,
        num_sizes: int = 100,
        warmup_iters: int = 10,
        measure_iters: int = 20,
        input_size_range: Tuple[int, int] = (224, 512),
        layer_nvml_interval_ms: float = 0.02,
    ):
        self.model_name = model_name.lower()
        self.algorithm = algorithm.lower()
        self.batch_size = batch_size
        self.warmup_iters = warmup_iters
        self.measure_iters = measure_iters
        self.layer_nvml_interval_ms = layer_nvml_interval_ms

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
        if self.algorithm in ALGORITHM_COMPATIBILITY:
            compatible_models = ALGORITHM_COMPATIBILITY[self.algorithm]
            if self.model_name not in compatible_models:
                raise ValueError(
                    f"Algorithm '{self.algorithm}' is only compatible with: "
                    f"{compatible_models}, not '{self.model_name}'"
                )

    def _load_model(self):
        if self.model_name not in MODEL_REGISTRY:
            available = list(MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unknown model: {self.model_name}. Available: {available}")

        print(f"\nLoading {self.model_name}...")
        self.model = MODEL_REGISTRY[self.model_name]()
        self.model.eval()
        print(f"  {self.model_name} loaded")

        cudnn_algorithms = [
            'gemm', 'implicit gemm', 'implicit precomp gemm',
            'fft', 'fft tiling', 'winograd', 'winograd nonfused',
        ]
        if self.algorithm in cudnn_algorithms:
            if not ai3.using_cudnn():
                raise RuntimeError(
                    f"Algorithm '{self.algorithm}' requires cuDNN, but cuDNN "
                    "is not available. Rebuild ai3 with USE_CUDNN flag."
                )

        print(f"  Applying ai3 '{self.algorithm}' algorithm conversion...")
        ai3.swap_conv2d(self.model, self.algorithm)

        conv_layers = sum(
            1 for _, m in self.model.named_modules()
            if isinstance(m, torch.nn.Conv2d) or hasattr(m, 'algorithm')
        )
        ai3_layers = sum(
            1 for _, m in self.model.named_modules()
            if hasattr(m, 'algorithm')
        )

        print(f"  Conversion completed:")
        print(f"    Total Conv2D layers: {conv_layers}")
        print(f"    AI3 converted layers: {ai3_layers}")
        if conv_layers > 0:
            print(f"    Conversion rate: {ai3_layers / conv_layers * 100:.1f}%")

    # ------------------------------------------------------------------
    # Overall measurement (unchanged from v1)
    # ------------------------------------------------------------------
    def _measure_overall(self, input_data: torch.Tensor) -> Dict[str, float]:
        with torch.inference_mode():
            for _ in range(self.warmup_iters):
                _ = self.model(input_data)
                if self.use_cuda:
                    torch.cuda.synchronize()

        times = []
        all_power_samples = []

        with torch.inference_mode():
            for _ in range(self.measure_iters):
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

                samples = self.power_monitor.stop_sampling()
                times.append(duration_ms)
                if samples:
                    all_power_samples.extend(samples)

        stats = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
        }

        if all_power_samples:
            power_stats = self.power_monitor.compute_energy_trapezoidal(
                all_power_samples)
            stats.update({
                'power_mean_w': power_stats['avg_power_w'],
                'power_std_w': power_stats['power_std_w'],
                'power_min_w': power_stats['power_min_w'],
                'power_max_w': power_stats['power_max_w'],
                'energy_total_j': power_stats['energy_j'],
                'energy_per_inference_j': (
                    power_stats['energy_j'] / self.measure_iters
                ),
                'power_samples': power_stats['num_samples'],
            })

        return stats

    # ------------------------------------------------------------------
    # Layer measurement (v2: continuous power sampling + slicing)
    # ------------------------------------------------------------------
    def _measure_layers(
        self,
        input_data: torch.Tensor,
        overall_stats: Dict[str, float],
    ) -> Tuple[Dict, Dict]:
        """
        Returns (timing_stats, power_stats) where power_stats contains
        per-layer power/energy from NVML (hybrid + optional proportional fallback).
        """
        self.timer.reset()

        with torch.inference_mode():
            for _ in range(self.warmup_iters):
                _ = self.model(input_data)
                if self.use_cuda:
                    torch.cuda.synchronize()

        self.timer.reset()

        self.power_monitor.start_continuous_sampling(
            interval_ms=self.layer_nvml_interval_ms)

        with torch.inference_mode():
            for _ in range(self.measure_iters):
                _ = self.model(input_data)
                if self.use_cuda:
                    torch.cuda.synchronize()

        all_samples = self.power_monitor.stop_continuous_sampling()

        timing_stats = self.timer.get_statistics()
        epi = overall_stats.get('energy_per_inference_j')
        power_stats = self.timer.get_power_statistics(
            self.power_monitor,
            all_samples,
            timing_stats,
            float(epi) if epi is not None else None,
            self.measure_iters,
        )

        return timing_stats, power_stats

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------
    def run(self, output_dir: str = "./results") -> Dict:
        print("=" * 70)
        print(
            f"UNIFIED PROFILER v2: "
            f"{self.model_name.upper()} + {self.algorithm.upper()}")
        print("=" * 70)

        print(f"\nConfiguration:")
        print(f"  Model: {self.model_name}")
        print(f"  Algorithm: {self.algorithm}")
        print(f"  Batch size: {self.batch_size}")
        print(
            f"  Input sizes: {len(self.input_sizes)} "
            f"({self.input_sizes[0]} to {self.input_sizes[-1]})")
        print(f"  Warmup iterations: {self.warmup_iters}")
        print(f"  Measurement iterations: {self.measure_iters}")
        print(
            f"  Per-layer NVML interval: {self.layer_nvml_interval_ms} ms "
            f"(requested; OS may quantize sleep)")

        self._check_compatibility()
        self._load_model()

        if self.use_cuda:
            print(f"\n  CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  cuDNN version: {torch.backends.cudnn.version()}")
        else:
            print("\n  CUDA not available, running on CPU")

        self.timer = CUDALayerTimer(use_cuda=self.use_cuda)
        self.timer.register_hooks(self.model)

        overall_results = []
        layer_results = []

        print(f"\n{'=' * 70}")
        print("STARTING PROFILING")
        print(f"{'=' * 70}")

        for idx, input_size in enumerate(self.input_sizes, 1):
            print(
                f"\n[{idx}/{len(self.input_sizes)}] "
                f"Input size: {input_size}x{input_size}")

            if self.use_cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            try:
                input_data = torch.randn(
                    self.batch_size, 3, input_size, input_size)
                print(f"  Input tensor: {tuple(input_data.shape)}")
            except Exception as e:
                print(f"  Error creating input: {e}")
                continue

            # --- overall ---
            try:
                overall_stats = self._measure_overall(input_data)
                print(
                    f"  Time: {overall_stats['mean']:.2f}ms "
                    f"+/- {overall_stats['std']:.2f}ms")

                if 'power_mean_w' in overall_stats:
                    print(
                        f"  Power: {overall_stats['power_mean_w']:.2f}W "
                        f"(trapezoidal, "
                        f"{overall_stats.get('power_samples', 'N/A')} samples)")
                    print(
                        f"  Energy: "
                        f"{overall_stats.get('energy_per_inference_j', 0):.4f}"
                        f"J per inference")

                overall_results.append({
                    'model': self.model_name,
                    'algorithm': self.algorithm,
                    'batch_size': self.batch_size,
                    'input_size': input_size,
                    **overall_stats,
                })
            except Exception as e:
                print(f"  Error during overall measurement: {e}")
                import traceback
                traceback.print_exc()
                continue

            # --- layers (v2: timing + power) ---
            try:
                layer_timing, layer_power = self._measure_layers(
                    input_data, overall_stats)
                layer_info = self.timer.get_layer_info()

                if layer_timing:
                    sorted_layers = sorted(
                        layer_timing.items(),
                        key=lambda x: x[1]['mean'],
                        reverse=True,
                    )[:3]
                    print("  Top 3 slowest layers:")
                    for layer_name, stats in sorted_layers:
                        pct = (stats['mean'] / overall_stats['mean']) * 100
                        print(
                            f"    {layer_name}: {stats['mean']:.2f}ms "
                            f"({pct:.1f}%)")

                    for layer_name, stats in layer_timing.items():
                        info = layer_info.get(layer_name, {})
                        pwr = layer_power.get(layer_name, {})
                        layer_results.append({
                            'model': self.model_name,
                            'algorithm': self.algorithm,
                            'input_size': input_size,
                            'layer': layer_name,
                            'in_channels': info.get('in_channels', 'N/A'),
                            'out_channels': info.get('out_channels', 'N/A'),
                            'kernel_size': _to_single(
                                info.get('kernel_size', 'N/A')),
                            'stride': _to_single(info.get('stride', 1)),
                            'padding': _to_single(info.get('padding', 0)),
                            'mean_ms': stats['mean'],
                            'std_ms': stats['std'],
                            'percentage': (
                                stats['mean'] / overall_stats['mean']
                            ) * 100,
                            'power_mean_w': pwr.get('power_mean_w', 0),
                            'power_std_w': pwr.get('power_std_w', 0),
                            'energy_per_layer_j': pwr.get(
                                'energy_per_layer_j', 0),
                            'power_samples': pwr.get('power_samples', 0),
                            'energy_attribution': pwr.get(
                                'energy_attribution', 'none'),
                        })
            except Exception as e:
                print(f"  Error during layer measurement: {e}")

            del input_data
            if self.use_cuda:
                torch.cuda.empty_cache()

        self.timer.remove_hooks()
        self.power_monitor.cleanup()

        self._save_results(overall_results, layer_results, output_dir)

        print(f"\n{'=' * 70}")
        print("PROFILING COMPLETED")
        print(f"{'=' * 70}")
        print(f"  {len(overall_results)} input sizes profiled")
        print(f"  {len(layer_results)} layer measurements")
        print(f"  Results saved to: {output_dir}")

        return {'overall': overall_results, 'layers': layer_results}

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------
    def _save_results(
        self,
        overall: List[Dict],
        layers: List[Dict],
        output_dir: str,
    ):
        os.makedirs(output_dir, exist_ok=True)

        algo_safe = self.algorithm.replace(' ', '_')
        base_name = f"{self.model_name}_{algo_safe}"

        if overall:
            overall_path = os.path.join(
                output_dir, f"{base_name}_overall.csv")
            with open(overall_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=overall[0].keys())
                writer.writeheader()
                writer.writerows(overall)
            print(f"\n  Overall results: {overall_path}")

        if layers:
            layers_path = os.path.join(
                output_dir, f"{base_name}_layers.csv")
            with open(layers_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=layers[0].keys())
                writer.writeheader()
                writer.writerows(layers)
            print(f"  Layer results: {layers_path}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def _to_single(val):
    """Convert tuple values to single integers for CSV output."""
    if isinstance(val, tuple):
        if len(val) == 2 and val[0] == val[1]:
            return val[0]
        elif len(val) == 2:
            return f"{val[0]}x{val[1]}"
        elif len(val) == 1:
            return val[0]
    return val


def print_cuda_info():
    """Print CUDA device information."""
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return False

    print("\nCUDA Device Information:")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  cuDNN version: {torch.backends.cudnn.version()}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")
    print(f"  Current GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU Memory: {props.total_memory / 1024**3:.1f} GB")
    print(f"  Compute Capability: {props.major}.{props.minor}")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    return True


def list_available_models():
    return list(MODEL_REGISTRY.keys())


def list_available_algorithms():
    return ALGORITHMS.copy()


def check_compatibility(model_name: str, algorithm: str) -> bool:
    if algorithm in ALGORITHM_COMPATIBILITY:
        return model_name in ALGORITHM_COMPATIBILITY[algorithm]
    return True

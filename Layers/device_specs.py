"""
GPU hardware specifications lookup table.

Uses numerical device features instead of one-hot encoding so that ML models
can generalize to unseen GPUs by interpolating on hardware characteristics.
"""

from typing import Dict

DEVICE_SPECS: Dict[str, Dict[str, float]] = {
    'h100': {
        'memory_bandwidth_gbps': 3350.0,
        'fp32_tflops': 67.0,
        'tensor_tflops_fp16': 2000.0,
        'memory_gb': 80.0,
        'cuda_cores': 16896,
        'tensor_cores': 528,
        'tdp_watts': 700.0,
        'arch_generation': 9.0,       # Hopper
        'compute_capability': 9.0,
    },
    'l40s': {
        'memory_bandwidth_gbps': 864.0,
        'fp32_tflops': 91.6,
        'tensor_tflops_fp16': 733.0,
        'memory_gb': 48.0,
        'cuda_cores': 18176,
        'tensor_cores': 568,
        'tdp_watts': 350.0,
        'arch_generation': 8.9,       # Ada Lovelace
        'compute_capability': 8.9,
    },
    'v100': {
        'memory_bandwidth_gbps': 900.0,
        'fp32_tflops': 15.7,
        'tensor_tflops_fp16': 125.0,
        'memory_gb': 16.0,
        'cuda_cores': 5120,
        'tensor_cores': 640,
        'tdp_watts': 300.0,
        'arch_generation': 7.0,       # Volta
        'compute_capability': 7.0,
    },
    'a100': {
        'memory_bandwidth_gbps': 2039.0,
        'fp32_tflops': 19.5,
        'tensor_tflops_fp16': 312.0,
        'memory_gb': 80.0,
        'cuda_cores': 6912,
        'tensor_cores': 432,
        'tdp_watts': 400.0,
        'arch_generation': 8.0,       # Ampere
        'compute_capability': 8.0,
    },
    'rtx4090': {
        'memory_bandwidth_gbps': 1008.0,
        'fp32_tflops': 82.6,
        'tensor_tflops_fp16': 660.0,
        'memory_gb': 24.0,
        'cuda_cores': 16384,
        'tensor_cores': 512,
        'tdp_watts': 450.0,
        'arch_generation': 8.9,       # Ada Lovelace
        'compute_capability': 8.9,
    },
    'rtx2080': {
        'memory_bandwidth_gbps': 448.0,
        'fp32_tflops': 10.1,
        'tensor_tflops_fp16': 80.5,
        'memory_gb': 8.0,
        'cuda_cores': 2944,
        'tensor_cores': 368,
        'tdp_watts': 215.0,
        'arch_generation': 7.5,       # Turing
        'compute_capability': 7.5,
    },
}

DEVICE_FEATURE_NAMES = list(next(iter(DEVICE_SPECS.values())).keys())


def get_device_features(device_name: str) -> Dict[str, float]:
    """
    Return numerical feature dict for a given device name.
    Raises KeyError if the device is unknown.
    """
    key = device_name.lower().replace(' ', '').replace('-', '')
    if key in DEVICE_SPECS:
        return DEVICE_SPECS[key]
    raise KeyError(
        f"Unknown device '{device_name}'. "
        f"Available: {list(DEVICE_SPECS.keys())}"
    )


def list_devices():
    return list(DEVICE_SPECS.keys())

#!/usr/bin/env python3
"""
CLI entry point for the v2 Unified GPU Performance Profiler.

Usage:
    python run_profile.py --model vgg16 --algorithm gemm --test
    python run_profile.py --model vgg16 --algorithm gemm --quick
    python run_profile.py --model vgg16 --algorithm gemm --num-sizes 100
    python run_profile.py --list-models
    python run_profile.py --list-algorithms
    python run_profile.py --check-cuda
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from profiler import (
    UnifiedProfiler,
    list_available_models,
    list_available_algorithms,
    check_compatibility,
    print_cuda_info,
)


def main():
    parser = argparse.ArgumentParser(
        description='Unified GPU Performance Profiler v2 for ai3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_profile.py --model vgg16 --algorithm gemm --test
  python run_profile.py --model vgg16 --algorithm gemm --quick
  python run_profile.py --model vgg16 --algorithm gemm
  python run_profile.py --list-models
  python run_profile.py --list-algorithms
        """,
    )

    parser.add_argument('--model', '-m', type=str,
                        help='Model to profile')
    parser.add_argument('--algorithm', '-a', type=str,
                        help='Convolution algorithm')

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--test', action='store_true',
                            help='Test mode: 1 input size, 2 iterations')
    mode_group.add_argument('--quick', action='store_true',
                            help='Quick mode: 10 input sizes, 5 iterations')

    parser.add_argument('--num-sizes', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--measure', type=int, default=20)
    parser.add_argument('--min-size', type=int, default=224)
    parser.add_argument('--max-size', type=int, default=512)
    parser.add_argument('--output-dir', '-o', type=str, default='./results')
    parser.add_argument(
        '--layer-nvml-interval-ms',
        type=float,
        default=0.02,
        metavar='MS',
        help='Milliseconds between NVML polls during per-layer measurement '
             '(smaller → more samples, more host CPU; default 0.02)',
    )

    parser.add_argument('--list-models', action='store_true')
    parser.add_argument('--list-algorithms', action='store_true')
    parser.add_argument('--check-cuda', action='store_true')

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for model in list_available_models():
            print(f"  - {model}")
        return 0

    if args.list_algorithms:
        print("Available algorithms:")
        for algo in list_available_algorithms():
            print(f"  - {algo}")
        print("\nNote: 'winograd' and 'winograd nonfused' only work "
              "with vgg16 (3x3 kernels)")
        return 0

    if args.check_cuda:
        print_cuda_info()
        return 0

    if not args.model or not args.algorithm:
        parser.error("--model and --algorithm are required")

    if args.layer_nvml_interval_ms <= 0:
        parser.error("--layer-nvml-interval-ms must be positive")

    model_name = args.model.lower()
    algorithm = args.algorithm.lower()

    available_models = list_available_models()
    if model_name not in available_models:
        print(f"Unknown model: {model_name}")
        print(f"  Available: {', '.join(available_models)}")
        return 1

    available_algos = list_available_algorithms()
    if algorithm not in available_algos:
        print(f"Unknown algorithm: {algorithm}")
        print(f"  Available: {', '.join(available_algos)}")
        return 1

    if not check_compatibility(model_name, algorithm):
        print(f"Algorithm '{algorithm}' is not compatible "
              f"with model '{model_name}'")
        return 1

    if args.test:
        num_sizes = 1
        warmup = 1
        measure = 2
        print("TEST MODE: 1 input size, 2 iterations")
    elif args.quick:
        num_sizes = 10
        warmup = 5
        measure = 10
        print("QUICK MODE: 10 input sizes, 10 iterations")
    else:
        num_sizes = args.num_sizes
        warmup = args.warmup
        measure = args.measure
        print(f"FULL MODE: {num_sizes} input sizes, {measure} iterations")

    try:
        profiler = UnifiedProfiler(
            model_name=model_name,
            algorithm=algorithm,
            batch_size=args.batch_size,
            num_sizes=num_sizes,
            warmup_iters=warmup,
            measure_iters=measure,
            input_size_range=(args.min_size, args.max_size),
            layer_nvml_interval_ms=args.layer_nvml_interval_ms,
        )
        profiler.run(output_dir=args.output_dir)
        return 0

    except KeyboardInterrupt:
        print("\n\nProfiling interrupted by user")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
CLI entry point for the Unified GPU Performance Profiler.

Usage:
    # Basic usage
    python run_profile.py --model vgg16 --algorithm gemm
    
    # Test mode (1 input size, quick verification)
    python run_profile.py --model vgg16 --algorithm gemm --test
    
    # Quick mode (10 input sizes)
    python run_profile.py --model vgg16 --algorithm gemm --quick
    
    # Full mode (100 input sizes, default)
    python run_profile.py --model vgg16 --algorithm gemm --num-sizes 100
    
    # Custom settings
    python run_profile.py --model densenet121 --algorithm direct \
        --num-sizes 50 --warmup 5 --measure 15 --output-dir ./my_results

Available models: vgg16, densenet121, resnet152, googlenet, alexnet, resnet50
Available algorithms: direct, gemm, implicit gemm, implicit precomp gemm, 
                      fft, fft tiling, winograd, winograd nonfused, guess
"""

import argparse
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from profiler import (
    UnifiedProfiler,
    list_available_models,
    list_available_algorithms,
    check_compatibility,
    print_cuda_info
)


def main():
    parser = argparse.ArgumentParser(
        description='Unified GPU Performance Profiler for ai3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick sanity check
  python run_profile.py --model vgg16 --algorithm gemm --test
  
  # 10-point quick sweep
  python run_profile.py --model vgg16 --algorithm gemm --quick
  
  # Full 100-point profiling
  python run_profile.py --model vgg16 --algorithm gemm
  
  # List available options
  python run_profile.py --list-models
  python run_profile.py --list-algorithms
        """
    )

    # Model and algorithm
    parser.add_argument('--model', '-m', type=str,
                        help='Model to profile (e.g., vgg16, densenet121)')
    parser.add_argument('--algorithm', '-a', type=str,
                        help='Convolution algorithm (e.g., gemm, direct, winograd)')

    # Mode shortcuts
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--test', action='store_true',
                           help='Test mode: 1 input size, 2 iterations (quick sanity check)')
    mode_group.add_argument('--quick', action='store_true',
                           help='Quick mode: 10 input sizes, 5 iterations')

    # Detailed configuration
    parser.add_argument('--num-sizes', type=int, default=100,
                        help='Number of input sizes to test (default: 100)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Warmup iterations (default: 10)')
    parser.add_argument('--measure', type=int, default=20,
                        help='Measurement iterations (default: 20)')
    parser.add_argument('--min-size', type=int, default=224,
                        help='Minimum input size (default: 224)')
    parser.add_argument('--max-size', type=int, default=512,
                        help='Maximum input size (default: 512)')

    # Output
    parser.add_argument('--output-dir', '-o', type=str, default='./results',
                        help='Output directory for results (default: ./results)')

    # Info commands
    parser.add_argument('--list-models', action='store_true',
                        help='List available models and exit')
    parser.add_argument('--list-algorithms', action='store_true',
                        help='List available algorithms and exit')
    parser.add_argument('--check-cuda', action='store_true',
                        help='Print CUDA info and exit')

    args = parser.parse_args()

    # Handle info commands
    if args.list_models:
        print("Available models:")
        for model in list_available_models():
            print(f"  - {model}")
        return 0

    if args.list_algorithms:
        print("Available algorithms:")
        for algo in list_available_algorithms():
            print(f"  - {algo}")
        print("\nNote: 'winograd' and 'winograd nonfused' only work with vgg16 (3x3 kernels)")
        return 0

    if args.check_cuda:
        print_cuda_info()
        return 0

    # Validate required arguments
    if not args.model or not args.algorithm:
        parser.error("--model and --algorithm are required")

    # Normalize inputs
    model_name = args.model.lower()
    algorithm = args.algorithm.lower()

    # Validate model
    available_models = list_available_models()
    if model_name not in available_models:
        print(f"✗ Unknown model: {model_name}")
        print(f"  Available: {', '.join(available_models)}")
        return 1

    # Validate algorithm
    available_algos = list_available_algorithms()
    if algorithm not in available_algos:
        print(f"✗ Unknown algorithm: {algorithm}")
        print(f"  Available: {', '.join(available_algos)}")
        return 1

    # Check compatibility
    if not check_compatibility(model_name, algorithm):
        print(f"✗ Algorithm '{algorithm}' is not compatible with model '{model_name}'")
        print(f"  (Winograd algorithms require 3x3 kernels, only available for vgg16)")
        return 1

    # Apply mode shortcuts
    if args.test:
        num_sizes = 1
        warmup = 1
        measure = 2
        print("⚡ TEST MODE: 1 input size, 2 iterations")
    elif args.quick:
        num_sizes = 10
        warmup = 5
        measure = 10
        print("🏃 QUICK MODE: 10 input sizes, 10 iterations")
    else:
        num_sizes = args.num_sizes
        warmup = args.warmup
        measure = args.measure
        print(f"📊 FULL MODE: {num_sizes} input sizes, {measure} iterations")

    # Create and run profiler
    try:
        profiler = UnifiedProfiler(
            model_name=model_name,
            algorithm=algorithm,
            batch_size=args.batch_size,
            num_sizes=num_sizes,
            warmup_iters=warmup,
            measure_iters=measure,
            input_size_range=(args.min_size, args.max_size)
        )

        profiler.run(output_dir=args.output_dir)
        return 0

    except KeyboardInterrupt:
        print("\n\n✗ Profiling interrupted by user")
        return 130
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

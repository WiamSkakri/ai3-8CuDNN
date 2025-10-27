#!/bin/bash
#SBATCH --job-name=googlenet_gemm_l40s
#SBATCH --output=googlenet_gemm_results_l40s/googlenet_gemm_l40s_%j.out
#SBATCH --error=googlenet_gemm_results_l40s/googlenet_gemm_l40s_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH -C gpul40s
#SBATCH --gres=gpu:1

# Job Information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Print GPU information
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Load required modules (adjust based on your HPC system)
# Uncomment and modify as needed for your system
# module purge
# module load cuda/12.1
# module load cudnn/9.0
# module load python/3.9

# Activate conda/virtual environment
# Replace with your actual environment name
echo "Activating ai3_8cudnn environment..."
source ~/.bashrc  # or ~/.bash_profile
conda activate ai3_8cudnn

# Verify environment
echo ""
echo "Python version:"
python --version
echo ""
echo "PyTorch CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"
echo ""
echo "ai3 cuDNN available:"
python -c "import ai3; print(ai3.using_cudnn())"
echo ""

# Navigate to working directory (where the script is located)
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
echo "Files in directory:"
ls -la googlenet_gemm.py 2>/dev/null || echo "  ⚠ WARNING: googlenet_gemm.py not found in current directory!"
echo ""

# Verify script exists
if [ ! -f "googlenet_gemm.py" ]; then
    echo "✗ ERROR: googlenet_gemm.py not found!"
    echo "Expected location: $(pwd)/googlenet_gemm.py"
    echo "Please ensure you submit the job from the directory containing googlenet_gemm.py"
    exit 1
fi

# Create results directory
RESULTS_DIR="googlenet_gemm_results_l40s"
echo "Creating results directory: $RESULTS_DIR"
mkdir -p $RESULTS_DIR
echo "✓ Results will be saved to: $(pwd)/$RESULTS_DIR"
echo ""

# Change to results directory for output files
cd $RESULTS_DIR

# Run the profiling script
echo "=========================================="
echo "Starting GoogLeNet GEMM Profiling..."
echo "=========================================="
echo ""

# Run script from parent directory, output will be saved in current (results) directory
python ../googlenet_gemm.py

# Check exit status
EXIT_CODE=$?
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Profiling completed successfully"
    echo ""
    echo "Results saved to: $SLURM_SUBMIT_DIR/$RESULTS_DIR"
    echo "Files created:"
    ls -lh *.csv 2>/dev/null || echo "  (no CSV files found)"
else
    echo "✗ Profiling failed with exit code: $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE



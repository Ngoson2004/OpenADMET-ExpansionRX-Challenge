module load miniconda3/23.5.2

# Set cache directories for both conda and pip
export CONDA_CACHE_DIR="/datasets/work/compchem/work/NGO014_SonNgo_workspace/.cache"
export CONDA_PKGS_DIRS="/datasets/work/compchem/work/NGO014_SonNgo_workspace/.conda/pkgs"
export PIP_CACHE_DIR="/datasets/work/compchem/work/NGO014_SonNgo_workspace/.cache/pip"
export PYTHONUSERBASE="/datasets/work/compchem/work/NGO014_SonNgo_workspace/.local"

# Create directories if they don't exist
mkdir -p $CONDA_CACHE_DIR
mkdir -p $CONDA_PKGS_DIRS
mkdir -p $PIP_CACHE_DIR
mkdir -p $PYTHONUSERBASE

# Configure conda directories
conda config --add envs_dirs /datasets/work/compchem/work/NGO014_SonNgo_workspace/.conda/envs
conda config --add pkgs_dirs /datasets/work/compchem/work/NGO014_SonNgo_workspace/.conda/pkgs

source activate chemprop-py311
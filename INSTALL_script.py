import subprocess
import sys


def check_cuda_availability():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        # Assuming CUDA is not available if torch cannot be imported
        return False


def install_packages(cuda_available):
    # Base URL for PyTorch Geometric wheels
    base_url = "https://data.pyg.org/whl/torch-2.2.0+"
    url_suffix = "cu121.html" if cuda_available else "cpu.html"
    full_url = base_url + url_suffix

    # Install packages from requirements.txt
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    # Install torch_geometric
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch_geometric"])

    # Install additional packages with the specified source
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "torch_scatter", "torch_sparse", "torch_cluster",
         "torch_spline_conv", "-f", full_url])


if __name__ == "__main__":
    # It might be necessary to ensure PyTorch is installed before this script runs,
    # or you can add a step here to install a default version of PyTorch first.

    cuda_available = check_cuda_availability()
    install_packages(cuda_available)

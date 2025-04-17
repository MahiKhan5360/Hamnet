from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Check GPU availability
try:
    !nvidia-smi
except:
    print("No GPU detected. Please enable GPU runtime: Edit → Notebook settings → Hardware accelerator → GPU → Save")

# Check CUDA version
!nvcc --version || echo "CUDA not available"

# Uninstall existing packages to avoid conflicts
!pip uninstall -y numpy torch torchvision monai einops tensorboard scikit-learn opencv-python torchaudio torchsummary tensorboard-data-server opencv-python-headless

# Install compatible packages
!pip install --no-cache-dir numpy==1.25.2 torch==2.4.1 torchvision==0.19.1 monai==1.4.0 einops==0.8.1 tensorboard==2.18.0 scikit-learn==1.5.2 opencv-python==4.10.0.84

# Verify installations
!pip list | grep -E 'numpy|torch|torchvision|monai|einops|tensorboard|scikit-learn|opencv-python'

# Create project directory
!mkdir -p /content/HAMNET
%cd /content/HAMNET

# Prevent Colab disconnection
from IPython.display import display, Javascript
def keep_alive():
    display(Javascript('''
        function click() {
            document.querySelector("colab-toolbar-button#connect").click()
        }
        setInterval(click, 60000)
    '''))
keep_alive()

# Verify CUDA
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("Running on the CPU. To use the GPU, enable it in Notebook settings.")

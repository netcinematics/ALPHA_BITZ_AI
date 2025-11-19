print("inside")
import subprocess
import sys

# is_colab = "google.colab" in sys.modules
is_kaggle = "kaggle_secrets" in sys.modules
print(is_kaggle)
# torch-scatter binaries depend on the torch and CUDA version, so we define the
# mappings here for Colab & Kaggle
# torch_to_cuda = {"1.10.0": "cu113", "1.9.0": "cu111", "1.9.1": "cu111"}


def install_requirements():
    print("⏳ Installing requirements ...")

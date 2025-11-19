import subprocess
import sys

# example code for kaggle import
is_kaggle = "kaggle_secrets" in sys.modules

def install_requirements():
    print("⏳ Installing requirements ...")

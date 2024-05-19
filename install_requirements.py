import subprocess
import sys

def install_package(command):
    try:
        subprocess.check_call([sys.executable, "-m"] + command)
        print(f"Successfully installed: {' '.join(command)}")
    except subprocess.CalledProcessError as e:
        print(f"Error installing {' '.join(command)}: {e}")
        sys.exit(1)

def install_requirements():
    packages = [
        ["pip", "install", "torch==2.2.1", "torchvision==0.17.1", "torchaudio==2.2.1", "--index-url", "https://download.pytorch.org/whl/rocm5.7"],
        ["pip", "install", "ultralytics"],
        ["pip", "install", "opencv-python-headless"],
        ["pip", "install", "matplotlib"]
    ]
    
    for package in packages:
        install_package(package)

if __name__ == "__main__":
    install_requirements()

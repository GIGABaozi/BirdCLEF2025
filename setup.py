from setuptools import setup, find_packages

setup(
    name="birdclef2025",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.5",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "librosa>=0.8.1",
        "timm>=0.5.4",
        "tqdm>=4.62.0",
        "matplotlib>=3.4.3",
        "seaborn>=0.11.2",
        "opencv-python>=4.5.3"
    ]
) 
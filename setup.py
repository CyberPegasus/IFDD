from setuptools import setup, find_packages
exec(open('core/version.py').read())

setup(
    name='IFDD',
    version=__version__,
    author='IFDD',
    description='Implementation Code for the paper "Lifting Scheme-Based Implicit Disentanglement of Emotion-Related Facial Dynamics in the Wild"',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'tqdm',
        'numpy',
        'opencv_python',
        'loguru',
        'tensorboard',
        'torch_tb_profiler',
        'torchmetrics',
        'ema_pytorch',
        'easydict',
        'albumentations',
        'scikit-learn',
        'scipy',
    ],
)
from setuptools import setup, find_packages

setup(
    name='my_package',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'fastapi==0.111.0',
        'einops==0.8.0',
        'torch==2.3.1',
        'torchvision==0.18.1',
        'matplotlib==3.9.0',
        'pandas==2.2.2',
        'numpy==1.26.4',
        'pillow==9.5.0',
        'scikit-learn==1.5.0',
    ],
)

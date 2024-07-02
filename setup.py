from setuptools import setup, find_packages

setup(
    name='my_package',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'fastapi==0.111.0'
        'einops==0.8.0'
        'torch==2.3.0'
        'torchvision==0.18.0'
        'scikit-image==0.19.3'
        'matplotlib==3.7.1'
        'pandas==2.0.3'
        'numpy==1.25.2'
        'pillow==9.4.0'
        'scikit-learn==1.2.2'
    ],
)
from setuptools import setup, find_packages

setup(
    name="langnav_robot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "ultralytics>=8.0.0",
        "clip-by-openai>=0.4.0",
        "stable-baselines3>=2.2.0",
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
    ],
    author="LangNav Team",
    description="Vision-language robot navigation with RL",
    license="MIT",
)

from setuptools import find_packages, setup

setup(
    name="multimodal_clinical_assistant",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[line.strip() for line in open("requirements.txt", encoding="utf-8") if line.strip()],
)

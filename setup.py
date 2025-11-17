from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="crypto-price-prediction",
    version="1.0.0",
    author="Votre Nom",
    author_email="votre.email@example.com",
    description="Système de prédiction de prix de cryptomonnaies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/votre-username/crypto-price-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.1.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.3.0",
        "tensorflow>=2.15.0",
        "fastapi>=0.104.0",
        "streamlit>=1.29.0",
    ],
)

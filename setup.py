from setuptools import setup, find_packages

setup(
    name="qsar-virtual-medchem",
    version="1.0.0",
    description="Virtual medicinal chemistry platform: QSAR, SAR, bioisosteres, drug-likeness",
    author="Harsh Nalgirkar",
    author_email="nalgirkarh@gmail.com",
    url="https://github.com/nalgirkarh-prog/qsar-virtual-medchem",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "scikit-learn>=1.3",
        "pandas>=2.0",
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "joblib>=1.3",
    ],
    entry_points={
        "console_scripts": [
            "medchem=medchem.medchem_cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)

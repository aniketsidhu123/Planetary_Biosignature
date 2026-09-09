from setuptools import setup, find_packages

setup(
    name="exoscope",
    version="0.1.0",
    description="Habitability & Biosignature-Likelihood Index (HBLI) — "
                "A composite, explainable scoring system for planetary habitability assessment",
    author="ExoScope Team",
    python_requires=">=3.10",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        "ultralytics>=8.3.0",
        "opencv-python-headless>=4.9.0",
        "scikit-learn>=1.4.0",
        "xgboost>=2.0.0",
        "shap>=0.44.0",
        "astroquery>=0.4.7",
        "requests>=2.31.0",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "Pillow>=10.0.0",
        "scikit-image>=0.22.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "plotly>=5.18.0",
        "streamlit>=1.38.0",
        "PyYAML>=6.0.0",
        "onnxruntime>=1.18.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",
        ],
        "labeling": [
            "labelImg>=1.8.6",
        ],
    },
    entry_points={
        "console_scripts": [
            "exoscope=src.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
    ],
)

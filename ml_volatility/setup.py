from setuptools import setup

# flake8: noqa

SERVICE = "ml_volatility"
NAME = f"{SERVICE}"

setup(
    name=NAME,
    version="0.1",
    description="Realized Volatility Prediction",
    long_description="It is a repository with all the code and documentation for the scientific work related to the volatility modeling using ML methods.",
    author="Iakov Grigoryev",
    author_email="Iakov Grigoryev <igrigoryev@nes.ru>",
    url="https://github.com/JacobGrig/ML-volatility",
    zip_safe=True,
    packages=[NAME],
    install_requires=[
        "torch",
        "numpy",
        "scipy",
        "pandas",
        "tqdm",
        "jupyter",
        "black",
        "scikit-learn",
        "matplotlib",
		"statsmodels",
    ],
    entry_points={"console_scripts": [f"{SERVICE}={NAME}.cli:cli_strategy"]},
    dependency_links=[],
)

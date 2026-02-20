from setuptools import setup, find_packages

setup(
    name="scpn-fusion",
    version="3.6.0",
    license="AGPL-3.0-or-later",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "click>=8.1",
        "numpy",
        "matplotlib",
        "scipy",
        "streamlit"
    ],
)

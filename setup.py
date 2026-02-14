from setuptools import setup, find_packages

setup(
    name="scpn-fusion",
    version="1.0.2",
    license="AGPL-3.0-or-later",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "streamlit"
    ],
)

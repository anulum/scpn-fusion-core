from setuptools import setup, find_packages

setup(
    name="scpn-fusion",
    version="2.0.0a1",
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

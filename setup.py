from setuptools import setup, find_packages

setup(
    name="scpn-fusion",
    version="3.9.1",
    license="AGPL-3.0-or-later",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "click>=8.1",
        "numpy<2.0",
        "matplotlib",
        "scipy",
        "streamlit",
        "jax<0.5.0",
        "jaxlib<0.5.0",
        "gymnasium>=1.0.0",
        "pydantic>=2.0",
        "pandas>=1.5",
    ],
)

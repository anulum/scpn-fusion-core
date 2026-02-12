# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Package Setup
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from setuptools import setup, find_packages

setup(
    author="Miroslav Sotek",
    author_email="protoscience@anulum.li",
    license="AGPL-3.0-or-later",
    url="https://github.com/anulum/scpn-fusion-core",
    name="scpn-fusion",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "streamlit"
    ],
)

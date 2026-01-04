from setuptools import setup, find_packages

setup(
    name="grandmasterism-v2",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "chess>=1.10.0",
        "qutip>=4.7.0",
        "numpy>=1.21.0"
    ],
    description="Grandmasterism v2 Pinnacle - Eternal quantum strategy",
    author="Eternally-Thriving-Grandmasterism",
    license="MIT",
)

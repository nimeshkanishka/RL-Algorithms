from setuptools import setup, find_packages

setup(
    name="rl_algorithms",
    version="0.1.1",
    author="Nimesh Kanishka",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "gymnasium"
    ]
)
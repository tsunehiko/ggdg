from setuptools import setup, find_packages

setup(
    name="ggdg",
    version="0.1",
    author='tsunehiko',
    description="grammar-based game description generation",
    packages=find_packages(
        exclude=["*_test.py", "test_*.py", "tests"]
    ),
)
from pathlib import Path
from setuptools import setup, find_packages

README = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="ai-trading-bot",
    version="0.1.0",
    description="Advanced self-improving AI day trading system",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Gervacius Lab",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[],
)

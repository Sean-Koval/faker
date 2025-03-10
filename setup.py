"""Setup script for the faker package."""

from setuptools import find_packages, setup

setup(
    name="faker",
    version="0.1.0",
    description="Modular synthetic chat data generation",
    author="Sean Koval",
    author_email="youremail@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "google-cloud-aiplatform",
        "python-dotenv",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
            "mypy",
        ],
    },
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "faker=faker.__main__:main",
        ],
    },
)

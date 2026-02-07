


from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh.readlines() if line.strip() and not line.startswith("#")]

setup(
    name="alphaearth-foundations",
    version="0.1.0",
    author="AlphaEarth Team",
    author_email="info@alphaearth.org",
    description="A foundation model for Earth observation data processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/alphaearth-foundations",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "alphaearth-train=src.alphaearth.run_train:main",
            "alphaearth-train-olmoearth=src.alphaearth.run_train_olmoearth_dataset:main",
        ],
    },
)
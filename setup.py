# -*- encoding: utf-8 -*-
import setuptools

with open("README.md", "r") as file:
    readme_contents = file.read()

setuptools.setup(
    name="roguewave",
    version="0.1.89",
    license="Apache 2 License",
    install_requires=[
        "pysofar>=0.1.13",
        "numpy",
        "netCDF4",
        "pandas",
        "scipy",
        "numba",
        "tqdm",
        "h5netcdf",
        "s3fs",
        "boto3",
        "xarray",
        "requests",
        "pygrib",
        "numba",
        "numba_progress",
    ],
    description="Python package to interact with Sofar wave data",
    long_description=readme_contents,
    long_description_content_type="text/markdown",
    author="Pieter Bart Smit",
    author_email="sofaroceangithubbot@gmail.com",
    url="https://github.com/sofarocean/roguewave.git",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    project_urls={"Sofar Ocean Site": "https://www.sofarocean.com"},
)

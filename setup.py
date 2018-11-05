import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

setuptools.setup(
    name="geepers_pkg",
    version="0.0.1",
    author="Andreas Nel",
    author_email="nel.andreas1@gmail.com",
    description="Package to create generation constructive hyper-heuristics.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AndreasNel/nid-with-gp-hh",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
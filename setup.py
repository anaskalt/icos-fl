#!/usr/bin/env python3
"""ICOS-FL Package Setup."""

import setuptools

def read_file(filename: str) -> str:
    """Read file content."""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().strip()

def get_requirements() -> list:
    """Parse requirements file."""
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f 
                if line.strip() and not line.startswith('#')]

setuptools.setup(
    name="icos-fl",
    version=read_file('VERSION'),
    author="Anastasios Kaltakis",
    author_email="anastasioskaltakis@gmail.com",
    description="ICOS Federated Learning Framework",
    long_description=read_file('README.md'),
    long_description_content_type="text/markdown",
    url="https://github.com/anaskalt/icos-fl",
    packages=setuptools.find_packages(),
    python_requires='>=3.12',
    install_requires=get_requirements(),
    entry_points={
        'console_scripts': [
            'icos-fl-server=icos_fl.server.server:main',
            'icos-fl-client=icos_fl.client.client:main',
        ],
    },
    include_package_data=True,
)
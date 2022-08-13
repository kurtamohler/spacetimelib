# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

# TODO: Decide which license to use
# with open('LICENSE') as f:
#    license = f.read()

setup(
    name='spacetime',
    version='0.0.0',
    description='Special relativity compute library',
    long_description=readme,
    author='Kurt Mohler',
    author_email='kurtamohler@gmail.com',
    url='https://github.com/kurtamohler/spacetimelib',
    # license=license,
    packages=find_packages(exclude=('images', 'old', 'examples', 'tests'))
)


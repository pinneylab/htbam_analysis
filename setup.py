from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='htbam_analysis',
    version='0.1.0',
    url='https://github.com/pinneylab/htbam_analysis',
    author='Duncan Muir',
    author_email='',
    description='Code for analysing biochemical assays run on MITOMI devices',
    packages=find_packages(),
    install_requires=requirements,
)

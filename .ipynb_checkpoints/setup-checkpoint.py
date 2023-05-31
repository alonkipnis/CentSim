from setuptools import setup

setup(name='centroid_similarity',
version='0.1.6',
description="multilabel classification using averaging and feature selection",
url='https://cs.idc.ac.il/~kipnis/',
author='Alon Kipnis',
author_email='alon.kipnis@runi.ac.il',
packages=['centroid_similarity'],
install_requires=['numpy', 'scipy', 'multiple-hypothesis-testing'],
)

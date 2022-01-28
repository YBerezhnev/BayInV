# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 18:55:57 2022

@author: BerezhnevYM
"""

import setuptools

setuptools.setup(
    name='src',                    # package name
    version='0.1',                          # version
    description='Implimentation of the Bayesian least square approach for velocity changes measurements',      # short description
    url='http://example.com',               # package URL
    install_requires=[ "numpy", "scipy", "numba", "matplotlib", "seaborn"],                    # list of packages this package depends
                                            # on.
    packages=setuptools.find_packages(),              # List of module names that installing
    )

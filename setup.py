
import setuptools

setuptools.setup(
    name='BayInV',                    # package name
    version='0.1',                          # version
    description='Implimentation of the Bayesian least square approach for velocity changes measurements',      # short description
    url='https://github.com/YBerezhnev/BayInV',               # package URL
    install_requires=[ "numpy", "scipy", "numba", "matplotlib", "seaborn"],                    # list of packages this package depends
                                            # on.
    packages=setuptools.find_packages(),              # List of module names that installing
    )

# setuptools
try:
    import setuptools
    print('Checking for setuptools... OK')
except:
    print('Checking for setuptools... NO')
    print('Error : Python package "setuptools" is required.')
    exit(0)


# ANNarchy
try:
    print('Checking for ANNarchy... ', end='')
    import ANNarchy
except:
    print('NO')
    print('Error : Python package "ANNarchy" is required.')
    print('For installation check: https://annarchy.readthedocs.io/en/latest/')
    exit(0)

dependencies = [
    'numpy',
    'scipy',
    'matplotlib',
    'cython',
    'sympy',
    'ANNarchy'
]

setuptools.setup(
    name="BGmodelSST",
    version="0.0.1",
    description="Package with main scripts of the basal ganglia model from Goenner, Maith, Koulouri, Baladron & Hamker (2020)",
    url="https://github.com/Olimaol/stopsignaltask_BG",
    packages=setuptools.find_packages(),
    install_requires=dependencies
)

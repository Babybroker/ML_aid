from setuptools import setup, find_packages

setup(
    name='mlaid',
    version='0.1.2',
    description='A python package providing forecasting functions for solar production',

    # url='https://github.com/EnergieID/entsoe-py',
    license='MIT',
    author='K. Dankers',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    # classifiers=[
    #     'Development Status :: 5 - Production/Stable',
    #
    #     # Indicate who your project is intended for
    #     'Intended Audience :: Developers',
    #     'Topic :: Scientific/Engineering',
    #
    #     # Pick your license as you wish (should match "license" above)
    #     'License :: OSI Approved :: MIT License',
    #
    #     # Specify the Python versions you support here. In particular, ensure
    #     # that you indicate whether you support Python 2, Python 3 or both.
    #     'Programming Language :: Python :: 3.9',
    #     'Programming Language :: Python :: 3.10',
    # ],

    keywords='Solar Forecast',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #py_modules=[""],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed.
    install_requires=['seaborn', 'xgboost', 'pandas>=1.4.0', 'scikit-learn'],

    include_package_data=True,
)

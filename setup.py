#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="nineml",
    version="0.1",
    package_data={'nineml': ['examples/AL/demos/*.py',
                             'examples/AL/sample_components/*.py']},
    packages=find_packages(),
    author="Andrew P. Davison, Eilif Muller, Mike Hull, Thomas G. Close",
    # add your name here if you contribute to the code
    author_email="tom.g.close@gmail.com",
    description=(
        "A tool for reading, writing and generally working with 9ML files."),
    long_description=open("README.rst").read(),
    license="BSD 3 License",
    keywords="computational neuroscience modeling interoperability XML",
    url="http://nineml.net",
    classifiers=['Development Status :: 4 - Beta',
                 'Environment :: Console',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Natural Language :: English',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 2',
                 'Topic :: Scientific/Engineering'],
    install_requires=['lxml', 'sympy'],
    tests_require=['nose']
)

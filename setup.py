# -*- coding: utf-8 -*-
"""
Setup file for automatic installation and distribution of AMfe.
Run: 'python setup.py sdist' for Source Distribution
Run: 'python setup.py install' for Installation
Run: 'python setup.py bdist_wheel' for Building Binary-Wheel
    (recommended for windows-distributions)
    Attention: For every python-minor-version an extra wheel has to be built
    Use environments and install different python versions by using
    conda create -n python34 python=3.4 anaconda
Run: 'pip install wheelfile.whl' for Installing Binary-Wheel
Run: 'python setup.py bdist --format=<format> für Binary-Distribution:
    <format>=gztar|ztar|tar|zip|rpm|pgktool|sdux|wininst|msi
    Recommended: tar|zip|wininst (evtl. msi)
Run: 'python setup.py bdist --help-formats' to find out which distribution
    formats are available
"""

# Uncomment next line for debugging
# DISTUTILS_DEBUG='DEBUG'

import sys


def query_yes_no(question, default="yes"):
    '''
    Ask a yes/no question and return their answer.

    Parameters
    ----------
    question: String
        The question to be asked

    default: String "yes" or "no"
        The default answer

    Returns:
    --------
    answer: Boolean
        Answer: True if yes, False if no.
    '''

    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no'.\n")


config = {
    'name': 'PYFETI',
    'version': '0.1',
    'description': 'Python FETI standalone library .',
    'long_description': 'Python FETI standalone library which provides data structure to solve Dual Domain Decompositon'
                        'methods and implement new versions ',
    'author': 'Guilherme Jenovencio',
    'url': 'https://gitlab.lrz.de/AM/pyfeti',
    'download_url': 'Where to download it.',
    'author_email': 'guilherme.jenovencio@tum.de',
    'maintainer': 'Guilherme Jenovencio',
    'maintainer_email': 'guilherme.jenovencio@tum.de',
    'install_requires': ['numpy>=1.10', 'scipy>=0.17', 'pandas', 'matplotlib', 'numdifftools','dill','mpi4py'],
    'tests_require': ['nose'],
    'packages': ['pyfeti'],
    'scripts': [],
    'entry_points': {},
    'provides': 'pyfeti',
    'platforms': 'Linux, Windows',
    'license': None
}


if __name__ == '__main__':
    from setuptools import setup
    setup(**config)

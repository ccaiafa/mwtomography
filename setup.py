from setuptools import setup, find_packages

from __init__ import __version__

setup(
    name='mwtomography',
    version=__version__,

    url='https://github.com/ccaiafa/mwtomography',
    author='Cesar Caiafa',
    author_email='ccaiafa@gmail.com',

    packages=find_packages(),
)
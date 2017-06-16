from setuptools import find_packages
from setuptools import setup

name = "pyMHT-simulator"
author = "Erik Liland"
author_email = "erik.liland@gmail.com"
description = "A framework for testing pyMHT"
license = "BSD"
keywords = 'simulation'
url = 'http://autosea.github.io/sf/2016/04/15/radar_ais/'
install_requires = ['matplotlib', 'numpy>=1.8.0', 'scipy', 'termcolor', 'seaborn']
dependency_links = ['https://github.com/erikliland/pyMHT/tarball/master']
packages = find_packages(exclude=['logs', 'data', 'profile'])

setup(
    name=name,
    author=author,
    author_email=author_email,
    description=description,
    license=license,
    keywords=keywords,
    packages=packages,
    include_package_data=True,
    install_requires=install_requires,
    dependency_links=dependency_links,
    entry_points={
        'console_scripts': [
            'simAllSingle=pysimulator.runScenarios:mainSingle',
            'simAllMulti=pysimulator.runScenarios:mainMulti'
        ]
    }
)

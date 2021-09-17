from setuptools import setup
from setuptools import find_packages

# Load the Readme file.
with open(file="README.md", mode="r") as readme_handle:
    long_description = readme_handle.read()

setup(
    name = 'lofi',
    author = 'Libor Kudela',
    author_email = 'LiborKudela@vutbr.cz',
    version = '0.1.0',
    description = 'A python MPI compatible optimization package',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url='https://github.com/LiborKudela/lofi',
    install_requires=[
        'numpy>=1.19.2',
        'pandas>=1.2.3',
        'DyMat>=0.7',
        'mpi4py>=3.0.3',
        'matplotlib>=3.1.0'],
    packages = find_packages(),
    keywords = 'optimization, Open Modelica, MPI',
    python_requires='>3.6.0',
    classifiers=[
        'Natural Language :: English',
        'Development Status :: 2 - Pre-Alpha'
        'Programing Language :: Python :: 3.6',
        'Programing Language :: Python :: 3.7',
        'Operating System :: POSIX :: Linux'
    ]
)

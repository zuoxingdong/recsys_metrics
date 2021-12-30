import os
from pathlib import Path

from setuptools import find_packages, setup


# only read through the raw file without importing any modules
def _get_version(dirpath):
    var_name = '__version__'
    file_path = Path(dirpath)/'__init__.py'
    with open(file_path, 'r') as fp:
        for line in fp:
            if line.startswith(var_name):
                g = {}
                exec(line, g)
                return g[var_name]
        raise ValueError(f'`{var_name}` must be defined within `{file_path.as_posix()}`')


# https://packaging.python.org/en/latest/discussions/install-requires-vs-requirements/
# NOTE: `install_requires` should be minimal, simplistic to run the project
# `requirements.txt` is for a full list of dependencies
setup(
    name='recsys_metrics',
    version=_get_version('recsys_metrics/'),
    author='Xingdong Zuo',
    url='https://github.com/zuoxingdong/recsys_metrics',
    keywords=['pytorch', 'metrics', 'machine learning', 'deep learning', 'recsys', 'recommender systems', 'information retrieval'],
    description='An efficient PyTorch implementation of the evaluation metrics in recommender systems.',
    packages=find_packages(exclude=['docs', 'tests']),
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.17.2',
        'torch>=1.3.1'
    ],
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        # Topics
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        # Supported OS
        "Operating System :: OS Independent",
        # Supported Python versions
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ]
)

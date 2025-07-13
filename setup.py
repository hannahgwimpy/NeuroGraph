from setuptools import setup, find_packages

setup(
    name='neurograph',
    version='0.0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # Add your project's dependencies here, e.g.:
        # 'numpy',
        # 'pandas',
    ],
    # Additional metadata is in pyproject.toml
)

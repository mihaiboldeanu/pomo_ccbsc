from setuptools import find_packages, setup
setup(
    name='pomo_ccbsc',
    packages=find_packages(),
    version='0.0.10',
    description='',
    author='',
    url="",
    license='MIT',
    platforms=["any"],
    install_requires=[
        "ccbsc-api==0.0.5",
        "numpy",
        "tensorflow",
        "matplotlib",
        "scikit-image",
    ],
    package_data={'': ['*.hdf5']},
    include_package_data=True
)

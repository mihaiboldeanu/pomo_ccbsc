from setuptools import find_packages, setup
setup(
    name='pomo_ccbsc',
    packages=find_packages(),
    version='0.0.2',
    description='',
    author='',
    url="",
    license='MIT',
    platforms=["any"],
    install_requires=[
        "ccbysc-api==0.0.4",
        "numpy",
        "tensorflow",
        "matplotlib",
        "scikit-image",
    ],
    package_data={'': ['*.hdf5']},
    include_package_data=True
)

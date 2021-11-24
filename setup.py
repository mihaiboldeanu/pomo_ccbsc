from setuptools import find_packages, setup
setup(
    name='pomo_ccbsc',
    packages=find_packages(),
    version='0.0.1',
    description='',
    author='',
    url="",
    license='MIT',
    platforms=["any"],

    install_requires=[
        "ccbysc-api==0.0.2",
        "numpy",
        "tensorflow==2.5.0",
        "matplotlib",
        "scikit-image",
    ]
)

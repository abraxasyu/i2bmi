import setuptools

with open("README.rst", "r") as handle:
    README = handle.read()

with open("requirements.txt") as handle:
    dependencies = list(handle.readlines())

setuptools.setup(
    name="i2bmi",
    version="0.2.4",
    install_requires = dependencies,
    packages=setuptools.find_packages(),
)

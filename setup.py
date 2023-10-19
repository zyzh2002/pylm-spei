import setuptools

with open("README.md", "r") as fh:
    description = fh.read()

setuptools.setup(
    name="pylm-spei",
    version="0.1.0",
    author="zyzh0",
    author_email="18255331+zyzh2002@users.noreply.github.com",
    packages=setuptools.find_packages(),
    description="Calculate SPEI from a 3D array of precipitation data with L-moments method and GEV distribution.",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/zyzh2002/pylm-spei",
    license='GPL-3.0',
    python_requires='>=3.10',
    install_requires=['numpy', 'numba','numba_stats'],
)
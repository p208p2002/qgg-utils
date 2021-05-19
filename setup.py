from setuptools import setup, setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name="qgg_utils",
    version='1.0',
    author='Philip Huang',
    author_email="p208p2002@gmail.com",
    description="qgg-scorer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/p208p2002/qgg-scorer",
    packages=setuptools.find_packages(),
    install_requires=[
       'transformers',
       'torch',
       'loguru',
       'geneticalgorithm',
       'numpy'
    ],
    package_data={'qgg_utils':['*.txt']},
    python_requires='>=3.6',
)

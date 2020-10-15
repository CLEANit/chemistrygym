from setuptools import setup, find_packages
import sys

setup(name='extractworld',
      packages=find_packages(),
      install_requires=[
          'gym',
          'matplotlib',
          'numpy',
          'cmocean'
      ],
      description='Implementation of extraction simulations in the OpenAI Gym environment framework.',
      author='CLEAN and UW ECE ML',
      url='https://github.com/CLEANit/ExtractWorld',
      version='0.0')
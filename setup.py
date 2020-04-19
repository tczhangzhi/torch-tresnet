from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
from torch_tresnet import __version__, __authors__
import sys

packages = find_packages()


def readme():
    with open('README.rst') as f:
        return f.read()


class PyTest(TestCommand):

    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ""

    def run_tests(self):
        import shlex
        # import here, cause outside the eggs aren't loaded
        import pytest

        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


setup(name='torch_tresnet',
      version=__version__,
      license='MIT',
      description="TResNet: High Performance GPU-Dedicated Architecture",
      long_description=readme(),
      packages=packages,
      url='https://github.com/tczhangzhi/torch-tresnet',
      author=__authors__,
      author_email='850734033@qq.com',
      keywords='',
      install_requires=['torch>=1.3', 'torchvision>=0.4.0', 'inplace-abn'])

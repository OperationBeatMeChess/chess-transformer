from setuptools import setup, find_packages

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='chess_transformer',
      version='0.0.0',
      description='Dual-headed ChessNet using a transformer backbone',
      url='https://github.com/OperationBeatMeChess/chess-transformer',
      author='Keith Gordon',
      author_email='keith.gordon9@gmail.com',
      license='MIT License',
      install_requires=['torch', 'timm', 'numpy'],
      long_description=long_description,
      long_description_content_type="text/markdown",
)

from setuptools import setup
from gdm._version import __version__

setup(name='gdm',
      version=__version__,
      description='Universal data model for profiling ocean gliders',
      url='http://github.com/kerfoot/gdm',
      author='John Kerfoot',
      author_email='jkerfoot@marine.rutgers.edu',
      license='GPL3.0',
      packages=['gdm'],
      python_requires='>=3.8.5',
      install_requires=[
            'gsw', 
            'netcdf4', 
            'numpy', 
            'pandas', 
            'pyyaml',
            'scipy', 
            'shapely', 
            'xarray'
      ],
      zip_safe=False)

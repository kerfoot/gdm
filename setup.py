from setuptools import setup

setup(name='gdm',
      version='0.0',
      description='',
      url='http://github.com/kerfoot/gdm',
      author='John Kerfoot',
      author_email='jkerfoot@marine.rutgers.edu',
      license='GPL3.0',
      packages=['gdm'],
      install_requires=[
            'gsw', 
            'netcdf4', 
            'numpy', 
            'pandas', 
            'scipy', 
            'shapely', 
            'pyyaml',
            'xarray'
      ],
      zip_safe=False)

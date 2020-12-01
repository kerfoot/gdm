# gdm
Platform agnostic data model for profiling ocean gliders.

## Introduction
**gdm** is a python software package for parsing raw glider data files into [pandas DataFrames](https://pandas.pydata.org/docs/reference/frame.html) and, ultimately, [xarray Datasets](http://xarray.pydata.org/en/stable/data-structures.html#dataset). The resulting dataset is a NetCDF-like data structure storing global attributes, variables and their associated attributes. The resulting data structures are inputs into a number of the processing packages and toolbox provided by the oceanographic community as well as provide simple methods to export the export the processed data sets to [CF](), [ACDD]() and [IOOS Metadata Profile]() compliant NetCDF files.

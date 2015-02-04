from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("model_cython.pyx"),
    include_dirs=[numpy.get_include()]  # hayleys_model.pyx
)

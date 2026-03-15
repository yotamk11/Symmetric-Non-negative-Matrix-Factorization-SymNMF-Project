from setuptools import setup, Extension

# Defining the extension module
module = Extension(
    "symnmf", 
    sources=["symnmfmodule.c", "symnmf.c"] 
)

setup(
    name="symnmf",
    version="1.0",
    description="Python interface for the SymNMF C extension",
    ext_modules=[module]
)
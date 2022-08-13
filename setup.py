from distutils.core import setup, Extension

setup(name='mykmeanssp',
      version='1.0',
      description='mykmeanssp for sp class',
      ext_modules=[Extension('mykmeanssp', sources=['kmeans.c'])])

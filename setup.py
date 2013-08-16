import os
from distutils.sysconfig import get_python_lib
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
  
numpy_inc = os.path.join(get_python_lib(plat_specific=1), 'numpy/core/include')
 
try: # do we have cython?
    from Cython.Distutils import build_ext
    setup(
      cmdclass = {'build_ext': build_ext},
      ext_modules = [Extension("cstagger", ["cstagger.pyx"],include_dirs=[numpy_inc])])

except ImportError: # try to continue with just the .c file
    setup(name='cstagger',
      ext_modules = [Extension("cstagger", ["cstagger.c"],include_dirs=[numpy_inc])])

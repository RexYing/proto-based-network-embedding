from distutils.core import setup, Extension

nautymodule = Extension('nauty', 
                        extra_objects = ['nauty.a'],
                        sources=['nautymodule.c'])

setup(name = "nauty", maintainer = "Rex Ying", maintainer_email = "rexying@stanford.edu",
      ext_modules = [nautymodule]
)


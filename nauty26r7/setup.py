import os
from distutils.core import setup, Extension
from distutils.sysconfig import get_config_vars

(opt,) = get_config_vars('OPT')
os.environ['OPT'] = " ".join(
        flag for flag in opt.split() if flag != '-Wstrict-prototypes'
)

nautymodule = Extension('nauty', 
                        extra_objects = ['nauty.a'],
                        sources=['nautymodule.cpp'])

setup(name = "nauty", maintainer = "Rex Ying", maintainer_email = "rexying@stanford.edu",
      ext_modules = [nautymodule]
)


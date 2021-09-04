from setuptools import setup, Extension
from torch.utils import cpp_extension
import torch

cpp_cuda = 'persist_param_cuda.cu' if torch.cuda.is_available() else 'persist_param_cpp.cc'

setup(name='checkpoint_cpp',
      ext_modules=[
            Extension(
                  name='handle_mutex',
                  sources=['handle_mutex.cc'],
                  language='c++'
            ), 
            cpp_extension.CppExtension(
                  name='persist_param', 
                  sources=['persist_param.cc', cpp_cuda, 'handle_mutex.cc'], 
                  include_dirs=['./parallel_hashmap'], 
                  library_dirs=['/usr/local/lib'], 
                  libraries=['pmemobj', 'pmem']
            )
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension}
)
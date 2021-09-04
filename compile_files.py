import os
import sys

os.chdir('./ckpt_manage')
os.system('python3 setup.py build')

python_version = sys.version[0:3]
pre_handle = 'handle_mutex'
pre_pp = 'persist_param'
after = '.cpython-'+python_version[0]+python_version[2]+'-x86_64-linux-gnu.so'
so_dir = './build/lib.linux-x86_64-'+python_version+'/'
handle = so_dir + pre_handle + after
pp = so_dir + pre_pp + after

os.system('cp -r '+ handle + ' ' + pp + ' .')
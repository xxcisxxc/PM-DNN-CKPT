import os
from multiprocessing import Process, Value

def ok_inc(is_ok):
    with is_ok.get_lock():
        is_ok.value += 1

def ok_wait(is_ok, val):
    while is_ok.value != val:
        continue

def apt_install(is_ok):
    os.system('apt update -y --force-yes')
    os.system('apt full-upgrade -y --force-yes')
    ok_inc(is_ok)
    os.system('apt autoremove -y --force-yes')
    os.system('apt install git-all -y --force-yes')
    ok_inc(is_ok)
    os.system('apt install make cmake build-essential pkg-config autoconf libndctl-dev libdaxctl-dev pandoc libfabric-dev -y --force-yes')
    ok_inc(is_ok)

def pip_install(is_ok):
    ok_wait(is_ok, 1)
    os.system('pip3 install "pybind11[global]"')
    os.system('pip3 install torch torchvision torchaudio')

def pmdk_install(is_ok):
    ok_wait(is_ok, 2)
    os.system('git clone https://github.com/pmem/pmdk')
    os.chdir('pmdk')
    ok_wait(is_ok, 3)
    os.system('make install')
    os.chdir('..')
    os.system('touch /etc/ld.so.conf.d/libpmemobj.conf')
    os.system('echo "/usr/local/lib/" > /etc/ld.so.conf.d/libpmemobj.conf')
    os.system('ldconfig')

if __name__ == '__main__':
    exec =  []
    is_ok = Value('i', 0)
    os.chdir('..')
    exec.append(Process(target = apt_install, args=(is_ok,)))
    exec.append(Process(target = pip_install, args=(is_ok,)))
    exec.append(Process(target = pmdk_install, args=(is_ok,)))
    for e in exec:
        e.start()
    for e in exec:
        e.join()
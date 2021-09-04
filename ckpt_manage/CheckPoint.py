import torch
from collections import OrderedDict
from time import time

from .persist_param import PersistParam, AllThreadEnd, complete_ckpted
from .handle_mutex import HandleMutex

def name_value(state_dict):
    return list(state_dict.keys()), list(state_dict.values())

class CKPTManage:
    def __init__(self, filename, named_params, state_dict=OrderedDict()):
        self.ckpt_name, value = name_value(state_dict)
        self.pp = PersistParam(filename, self.ckpt_name, value)
        self.param_name = [tup[0] for tup in named_params]
        self.handlers = [HandleMutex() for i in range(len(self.param_name))]
        self.CoW_flag = {}
        self._init_CoW()
        self.ckpt_ornot = False

        self.param_grad_name = []
        self.handler_grad = []
        self.Impact = []

    def _init_CoW(self):
        for n in self.param_name:
            self.CoW_flag[n] = False

    def empty_grad(self):
        if self.ckpt_ornot:
            self.param_grad_name = []
            self.handler_grad = []
            self.Impact = []
    
    def append(self, i):
        if self.ckpt_ornot: 
            self.param_grad_name.append(self.param_name[i])
            self.handler_grad.append(self.handlers[i])
            self.Impact.append(True)

    def allow_ckpt(self):
        self.ckpt_ornot = complete_ckpted()
        return self.ckpt_ornot

    def begin_ckpt(self, i, param):
        name, handle = self.param_grad_name[i], self.handler_grad[i]
        if self.ckpt_ornot:
            self.pp.create_setval_thread(name, param, handle, self.CoW_flag[name])
            handle.P_CKPT()
        else:
            if self.Impact[i]:
                if handle.TRY_P_CKPT():
                    print(name, "CoW?", self.CoW_flag[name])
                    t = time()
                    self.pp.change_base(name)
                    if (time() - t > 0.01):
                        self.CoW_flag[name] = True
                    else:
                        self.CoW_flag[name] = False
                else:
                    self.CoW_flag[name] = False
                self.Impact[i] = False
    
    def end_ckpt(self, i):
        if self.ckpt_ornot:
            self.handler_grad[i].V_UPDT()

    def wait_end(self):
        AllThreadEnd()
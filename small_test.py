import torch, time
from torchvision import models
"""
from persist_param import PersistParam, AllThreadEnd, complete_ckpted
from handle_mutex import HandleMutex

names = ['1', '2', '3']
values = [torch.tensor(0), torch.tensor([1., 2.]), torch.randn(3, 4)]
handles = [HandleMutex() for i in range(len(names))]
pp = PersistParam('ckpt.nvm', names, values)

for i in range(len(names)):
    handles[i].P_CKPT()
    pp.create_setval_thread(names[i], values[i], handles[i])
    values[i].add_(3)
    handles[i].V_UPDT()
AllThreadEnd()
print([torch.equal(values[i].cpu(), pp.getvalue(n)) for i, n in enumerate(names)])
"""
#from ckpt_manage.CheckPoint import CheckPoint
from ckpt_manage import CKPTManage

model = models.resnet18()

ckpt = CKPTManage('ckpt.nvm', model.named_parameters(), model.state_dict())
#named_param = []
#for tup in model.named_parameters():
#    if tup[0] == 'layer4.0.conv2.weight' or tup[0] == 'layer4.1.conv1.weight' or tup[0] == 'layer4.1.conv2.weight':
#        named_param.append((tup[0], tup[1]))

#ckpt = CheckPoint('ckpt.nvm', named_param, model.state_dict())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sd = model.state_dict()
name = [tup[0] for tup in model.named_parameters()]
#name = [tup[0] for tup in named_param]
value = [sd[n].clone().to(device) for n in name]
#for i in range(len(name)):
#    print(name[i], value[i].element_size(), value[i].size())

iteration = 20
value_ = []

t00 = time.time()
cur_ckpt_iter = 0
for i in range(iteration):
    time.sleep(0.02)
    if ckpt.allow_ckpt():
        cur_ckpt_iter = i
        value_ = []
    ckpt.empty_grad()
    print("Current Iteration: %d / Checkpoint Iteration: %d" % (i, cur_ckpt_iter))
    for j, n in enumerate(name):
        v = value[j]
        ckpt.append(j)
        ckpt.begin_ckpt(j, v)
        v.add_(5)
        if i == cur_ckpt_iter:
            value_.append(v.clone())
        ckpt.end_ckpt(j)

ckpt.wait_end()
t11 = time.time()
print("time %f" % (t11-t00))

#print(value_[i].cpu())
#print(ckpt.pp.getvalue(name[-1]))
ass = [torch.equal(value_[i].cpu(), ckpt.pp.getvalue(n)) for i, n in enumerate(name)]
print(ass)
print(not False in ass)
for i, a in enumerate(ass):
    if not a:
        print(name[i])
        print(sd[name[i]])
        print(value_[i].cpu())
        print(ckpt.pp.getvalue(name[i]))
        break
#print([torch.equal(value[i].cpu(), ckpt.pp.getvalue(n)) for i, n in enumerate(name)])
#print([(value[i].cpu(), ckpt.pp.getvalue(n)) for i, n in enumerate(name)])
#print(value)
#print(ckpt.pp.getvalue(name[-1]))
#time.sleep(10)
#print(ckpt.pp.getvalue(name[-1]))

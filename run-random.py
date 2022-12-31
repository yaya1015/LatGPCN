import sys
import os
import copy
import json
import datetime
import time

opt = dict()


def generate_command(opt):
    cmd = 'python test_latgpcn.py'
    for opt, val in opt.items():
        cmd += ' --' + opt + ' ' + str(val)
    return cmd

def run(opt):
    opt_ = copy.deepcopy(opt)
    os.system(generate_command(opt_))



ptb_rate = [0.2,0.4,0.6,0.8,1.0]
opt['attack'] = "random"
f = open('result.txt','r+')
f.read()
f.write('random----------------------'+"\n")
f.close()

f = open('result.txt','r+')
f.read()
f.write('cora:')
f.close()
opt['dataset'] = "cora" 
opt['lamda1']= 1.
opt['gamma'] = 0.5
opt['weight_decay']=1e-4
for pr in ptb_rate:
    opt['ptb_rate'] = pr
    run(opt)
f = open('result.txt','r+')
f.read()
f.write("\n")
f.close()

f = open('result.txt','r+')
f.read()
f.write('citeseer:')
f.close()
opt['dataset'] = "citeseer"
opt['lamda1'] = 10.
opt['weight_decay']=1e-3
opt['gamma'] = 1.5
for pr in ptb_rate:
    opt['ptb_rate'] = pr
    run(opt)
f = open('result.txt','r+')
f.read()
f.write("\n")
f.close()

f = open('result.txt','r+')
f.read()
f.write('pubmed:')
f.close()
opt['dataset'] = "pubmed"
opt['weight_decay']=1e-4
opt['lamda1'] = 10.
opt['gamma'] = 1.5
for pr in ptb_rate:
    opt['ptb_rate'] = pr
    run(opt)
f = open('result.txt','r+')
f.read()
f.write("\n")
f.close()

f = open('result.txt','r+')
f.read()
f.write('cora_ml:')
f.close()
opt['dataset'] = "cora_ml"
opt['lamda1']= 1
opt['gamma'] = 1.0
opt['weight_decay']=5e-5
for pr in ptb_rate:
    opt['ptb_rate'] = pr
    run(opt)
f = open('result.txt','r+')
f.read()
f.write("\n")
f.close()

f = open('result.txt','r+')
f.read()
f.write('wiki:')
f.close()
opt['dataset'] = "wiki"
opt['lamda1']= 0.1
opt['gamma'] = 5
opt['weight_decay']= 5e-5
for pr in ptb_rate:
    opt['ptb_rate'] = pr
    run(opt)
f = open('result.txt','r+')
f.read()
f.write("\n")
f.close()

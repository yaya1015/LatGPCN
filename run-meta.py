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



ptb_rate = [0.05,0.1,0.15,0.2,0.25]
opt['attack'] = "meta"
f = open('result.txt','r+')
f.read()
f.write('meta----------------------'+"\n")
f.close()

f = open('result.txt','r+')
f.read()
f.write('cora:')
f.close()
opt['dataset'] = "cora"
opt['lamda1']= 2
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
opt['lamda1'] = 10
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
opt['lamda1']= 10
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
opt['gamma'] = 8
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
for pr in ptb_rate:
    opt['ptb_rate'] = pr
    run(opt)
f = open('result.txt','r+')
f.read()
f.write("\n")
f.close()




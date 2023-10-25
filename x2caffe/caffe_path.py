import os
import sys

envroot = os.environ.get('MCHOME', os.environ['PWD'])
sys.path.append(envroot + '/toolchain/caffe/python')
sys.path.append(envroot + '/toolchain/caffe/python/caffe')

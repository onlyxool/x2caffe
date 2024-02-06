import os
import sys

envroot = os.environ.get('CAFFE_INSTALL_PREFIX', os.environ['PWD'])
sys.path.append(envroot + '/python')
sys.path.append(envroot + '/python/caffe')

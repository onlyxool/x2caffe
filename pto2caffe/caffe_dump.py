import os
import sys
envroot = os.environ.get('MCHOME', os.environ['PWD'])
sys.path.append(envroot + '/toolchain/caffe/python')
sys.path.append(envroot + '/toolchain/caffe/python/caffe')
import caffe
import numpy as np
from caffe.proto import caffe_pb2
from google.protobuf import text_format


def load_input_from_input_tensor(caffe_net, param_net, input_tensor):
    for i, input_name in enumerate(caffe_net.inputs):
        caffe_net.blobs[input_name].data[0, ...] = input_tensor


def dump_caffe_model(caffe_name, caffe_path, input_tensor, input_file_name):
    proto_file = caffe_path + '/' + caffe_name + '.prototxt'
    model_file = caffe_path + '/' + caffe_name + '.caffemodel'
    caffe_net = caffe.Net(proto_file, caffe.TEST, weights=model_file)

    param_net = caffe_pb2.NetParameter()
    text_format.Merge(open(proto_file).read(), param_net)

#    load_input_from_input_tensor(caffe_net, param_net, input_tensor)

    dump_path = envroot + '/dump/' + caffe_name +'/caffe/' + input_file_name
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    for i in range(len(caffe_net.layers)):
        caffe_net._forward(i, i)
        for idx in caffe_net._top_ids(i):
            data = caffe_net._blobs[idx].data
            file_size = caffe_net._blobs[idx].count
            file_name = dump_path + '/l' + '%03d'%i + 'o' + '_' + caffe_net._layer_names[i] + '_'+ caffe_net._blob_names[idx] +'_'+ str(file_size) + '.txt'
            np.savetxt(file_name, data.reshape(-1), fmt='%-10.6f')

        for idx in caffe_net._bottom_ids(i):
            data = caffe_net._blobs[idx].data
            file_size = caffe_net._blobs[idx].count
            file_name = dump_path + '/l' + '%03d'%i + 'i' + '_' + caffe_net._layer_names[i] + '_'+ caffe_net._blob_names[idx] +'_'+ str(file_size) + '.txt'
            np.savetxt(file_name, data.reshape(-1), fmt='%-10.6f')

        for j in range(len(caffe_net.layers[i].blobs)):
            data = caffe_net.layers[i].blobs[j].data
            file_size = caffe_net.layers[i].blobs[j].count
            file_name = dump_path + '/l' + '%03d'%i + 'i' + '_' + caffe_net._layer_names[i] + '_'+ '_'+ str(file_size) + '.txt'
            np.savetxt(file_name, data.reshape(-1), fmt='%-10.6f')

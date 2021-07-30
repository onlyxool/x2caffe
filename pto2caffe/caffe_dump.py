import os
import sys
envroot = os.environ.get('MCHOME', os.environ['PWD'])
sys.path.append(envroot + 'toolchain/caffe/python')
sys.path.append(envroot + 'toolchain/caffe/python/caffe')
import caffe
import numpy as np
from caffe.proto import caffe_pb2
from google.protobuf import text_format

sys.path.append(os.getenv('MCHOME') + 'toolchain/caffe2npu')
from npu_load_input import Input


def _get_proto_layer(param_net, layer_name):
    index = -1
    proto_layers = param_net.layer
    for i in range(len(proto_layers)):
        if len(proto_layers[i].include) != 0 and proto_layers[i].include[0].phase != caffe_pb2.TEST:
            continue
        if proto_layers[i].name == layer_name:
            return proto_layers[i]
    return None


def load_input_from_input_proto(caffe_net, param_net):
    for i, input_name in enumerate(caffe_net.inputs):
        layer_param = _get_proto_layer(param_net, input_name)
        if layer_param.type != 'Input':
            continue

        input_data = Input(caffe_net, input_name)
        bin_param = layer_param.bin_data_param
        dtype_list = ['u8', 's16', 'f32']
        input_data.set_source(bin_param.root_folder, bin_param.source, dtype=dtype_list[bin_param.data_format], shape=bin_param.shape.dim, sb=bin_param.shift_bit)
        trans_param = layer_param.transform_param
        crop_size = [trans_param.crop_size, trans_param.crop_size] if trans_param.crop_size > 0 else [trans_param.crop_h, trans_param.crop_w]
        input_data.set_param(trans_param.scale, trans_param.mean_value, trans_param.mean_file, crop_size)
        input_data.load_one_batch(caffe_net)


def load_input_from_input_tensor(caffe_net, param_net, input_tensor):
    for i, input_name in enumerate(caffe_net.inputs):
        caffe_net.blobs[input_name].data[0, ...] = input_tensor


def dump_caffe_model(caffe_name, caffe_path, input_tensor, input_file_name):
    proto_file = caffe_path + '/' + caffe_name + '.prototxt'
    model_file = caffe_path + '/' + caffe_name + '.caffemodel'
    caffe_net = caffe.Net(proto_file, caffe.TEST, weights=model_file)

    param_net = caffe_pb2.NetParameter()
    text_format.Merge(open(proto_file).read(), param_net)

#    load_input_from_input_proto(caffe_net, param_net)
    load_input_from_input_tensor(caffe_net, param_net, input_tensor)

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

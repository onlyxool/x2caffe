#import os
#import sys
#envroot = os.environ.get('MCHOME', os.environ['PWD'])
#sys.path.append(envroot + 'toolchain/caffe2npu')
#from compare_data import *
#from load_data import LoadData
#
#def compare(platform, model_name, input_file_name):
#    print('Comparing...')
#    dump_path = envroot + '/dump/' + model_name
#    target_data = dump_path +'/'+ platform +'/'+ input_file_name
#    caffe_data =  dump_path + '/caffe/' + input_file_name
#    data_load = LoadData()
#
#    if data_load.load_file(target_data, caffe_data) == False:
#        print("Input directories are not match, please check")
#
#    while 1:
#        file1, file2 = data_load.get_one_batch()
#        if file1 == None or file2 == None:
#            print('Dump File not found.')
#            break
#
#        array1 = np.asarray(read_file(file1))
#        array2 = np.asarray(read_file(file2))
#        if len(array1) != len(array2):
#            print("Get data error while calculate cosine similarity, lenght of 2 arrays are not equire!")
#            continue
#
#        cosin_simi = calc_cosine_simi(array1, array2) # close 1 is better
#        max_err = calc_max_diff(array1, array2) # close 0 is better
#        max_ratio = calc_max_ratio(array1, array2) # close 0 is better
#
#        layer_name = os.path.split(file1)[1].split('_')[1]
#        print(layer_name, ':')
#        print('  cosin_simi: %8f'% cosin_simi)
#        print('  cmax_err: %8f'% max_err)
#        print('  max_ratio: %8f'% max_ratio, '\n')

import os
import sys
envroot = os.environ.get('MCHOME', os.environ['PWD'])
sys.path.append(envroot + 'toolchain/caffe/python')
sys.path.append(envroot + 'toolchain/caffe/python/caffe')
import caffe
#import numpy as np
from caffe.proto import caffe_pb2
from google.protobuf import text_format

import math

def calc_cosine_simi(array1, array2):
    square_sum1 = sum(array1 ** 2)
    square_sum2 = sum(array2 ** 2)
    dot_product = sum(array1 * array2)

    result = dot_product / (math.sqrt(square_sum1) * math.sqrt(square_sum2))
    return result


def calc_max_diff(array1, array2):
    max_diff = 0.
    array = map(abs, array1 - array2)
    temp = max(array)

    max_diff = max(temp, max_diff)
    return max_diff


def calc_max_ratio(array1, array2):
    max_ratio = 0.
    sum1 = sum(array1)
    sum2 = sum(array2)
    temp = abs((sum1 - sum2) * 100 / sum2)

    max_ratio = max(temp, max_ratio)
    return max_ratio


def dump_caffe_model(caffe_name, caffe_path, input_tensor):
    proto_file = caffe_path + '/' + caffe_name + '.prototxt'
    model_file = caffe_path + '/' + caffe_name + '.caffemodel'
    caffe_net = caffe.Net(proto_file, caffe.TEST, weights=model_file)

    param_net = caffe_pb2.NetParameter()
    text_format.Merge(open(proto_file).read(), param_net)

    for i, input_name in enumerate(caffe_net.inputs):
        caffe_net.blobs[input_name].data[0, ...] = input_tensor

    caffe_output_dict = dict()
    blob_layer_map = dict()
    for i in range(len(caffe_net.layers)):
        caffe_net._forward(i, i)
        for idx in caffe_net._top_ids(i):
            data = caffe_net._blobs[idx].data
            caffe_output_dict[caffe_net._blob_names[idx]] = data.reshape(-1)
            blob_layer_map[caffe_net._blob_names[idx]] = caffe_net._layer_names[i]

    return caffe_output_dict, blob_layer_map


def compare(platform, target_model, caffe_model, caffe_path, input_tensor):
    print('Comparing...')

    caffe_output_dict, blob_layer_map = dump_caffe_model(caffe_model, caffe_path, input_tensor)

    if platform == 'tflite':
        from tflite2caffe.tflite_run import get_output
    elif platform == 'onnx':
        from onnx2caffe.onnx_run import get_output
    else:
        raise NotImplementedError(paltform)

    for blob_name in caffe_output_dict:
        target_output = get_output(target_model, input_tensor, blob_name)

        if target_output is None:
            continue
        else:
            target_output = target_output.reshape(-1)

        cosin_simi = calc_cosine_simi(caffe_output_dict[blob_name], target_output)
        max_err = calc_max_diff(caffe_output_dict[blob_name], target_output)
        max_ratio = calc_max_ratio(caffe_output_dict[blob_name], target_output)

        print(blob_layer_map[blob_name], ':')
        print('  cosin_simi: %8f'% cosin_simi)
        print('  cmax_err: %8f'% max_err)
        print('  max_ratio: %8f'% max_ratio, '\n')

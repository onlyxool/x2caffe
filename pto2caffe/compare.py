import os
import sys
import math
envroot = os.environ.get('MCHOME', os.environ['PWD'])
sys.path.append(envroot + 'toolchain/caffe/python')
sys.path.append(envroot + 'toolchain/caffe/python/caffe')

import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format


def calc_size(array1, array2):
    if array1.shape != array2.shape:
        return [array1.shape, array2.shape]
    else:
        return None


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


def dump_caffe_model(caffe_name, caffe_path, input_tensor, level):
    proto_file = caffe_path + '/' + caffe_name + '.prototxt'
    model_file = caffe_path + '/' + caffe_name + '.caffemodel'
    caffe_net = caffe.Net(proto_file, caffe.TEST, weights=model_file)

    param_net = caffe_pb2.NetParameter()
    text_format.Merge(open(proto_file).read(), param_net)

#    for i, input_name in enumerate(caffe_net.inputs):
#        caffe_net.blobs[input_name].data[0, ...] = input_tensor

    caffe_output_dict = dict()
    blob2layer_map = dict()

    if level == 1:
        caffe_net._forward(0, len(caffe_net.layers)-1)
        for idx in caffe_net._top_ids(len(caffe_net.layers)-1):
            data = caffe_net._blobs[idx].data
            caffe_output_dict[caffe_net._blob_names[idx]] = data.reshape(-1)
            blob2layer_map[caffe_net._blob_names[idx]] = caffe_net._layer_names[len(caffe_net.layers)-1]
    elif level == 2:
        for i in range(len(caffe_net.layers)):
            caffe_net._forward(i, i)
            for idx in caffe_net._top_ids(i):
                data = caffe_net._blobs[idx].data
                caffe_output_dict[caffe_net._blob_names[idx]] = data.reshape(-1)
                blob2layer_map[caffe_net._blob_names[idx]] = caffe_net._layer_names[i]

    return caffe_output_dict, blob2layer_map


def compare(platform, target_model, caffe_model, caffe_path, input_tensor, level=-1):
    if level <= 0:
        return
    else:
        print('Comparing...')

    caffe_output_dict, blob2layer_map = dump_caffe_model(caffe_model, caffe_path, input_tensor, level)

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

        print(blob2layer_map[blob_name], '[', blob_name, ']:')
        size_diff = calc_size(caffe_output_dict[blob_name], target_output)
        if size_diff is not None:
            print('  Size compare error:', caffe_output_dict[blob_name].shape, target_output.shape)

        cosin_simi = calc_cosine_simi(caffe_output_dict[blob_name], target_output)
        max_err = calc_max_diff(caffe_output_dict[blob_name], target_output)
        max_ratio = calc_max_ratio(caffe_output_dict[blob_name], target_output)

#        if (cosin_simi - 1) > 0.02 or (1 - cosin_simi) > 0.02:
#            print(platform, target_output.dtype, target_output, target_output.shape)
#            print('caffe', caffe_output_dict[blob_name].dtype, caffe_output_dict[blob_name], caffe_output_dict[blob_name].shape)

        print('  cosin_simi: %8f'% cosin_simi)
        print('  cmax_err: %8f'% max_err)
        print('  max_ratio: %8f'% max_ratio, '\n')

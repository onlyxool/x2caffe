import os
import sys
import cv2
import torch
import random
import argparse
import numpy as np

#sys.path.insert(0, '../framework/python')
#sys.path.insert(0, '../framework/python/caffe')
#import caffe


#import PytorchConvert as PytorchConvert
#import TensorflowConvert as TensorflowConvert

trans_dtype = {
    'u8': np.uint8,
    's8': np.int8,
    's16': np.int16,
    'f32': np.float32
}

def RGB2BGR(param):
    if isinstance(param, list) and len(param) == 3:
        ret_param = []
        ret_param.append(param[2])
        ret_param.append(param[1])
        ret_param.append(param[0])
        return ret_param
    else:
        return param

def get_input_shape(model_path):
    model_name, ext = model_path.split('.')
    shape = []

    if ext == 'onnx':
        import onnx
        model = onnx.load(model_path)
        graph = model.graph
        for dim in graph.input[0].type.tensor_type.shape.dim:
            shape.append(dim.dim_value)
        if shape[0] == 0:
            shape[0] = 1
    else:
        raise NotImplementedError

    return shape

def get_bin(param, path):
    bin_data = np.fromfile(path, trans_dtype[param['dtype']])
    bin_data = np.array(bin_data, dtype=np.float32)

    shape = get_input_shape(param['model'])
    bin_data = np.reshape(bin_data, shape)

    return bin_data

def get_image(param, path):
    image = cv2.imread(path)
    assert(image is not None)

    if param['color_format'] == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if param['new_height'] is not None and param['new_width'] is not None:
        new_h = param['new_height']
        new_w = param['new_width']
        image = cv2.resize(image, (new_h, new_w), interpolation = cv2.INTER_CUBIC)

    if param['crop_h'] is not None and param['crop_w'] is not None:
        crop_h = param['crop_h']
        crop_w = param['crop_w']
        if image.shape[0] <= crop_h or image.shape[1] <= crop_w:
            raise ValueError('crop size > image size, image:', image.shape, 'crop:', [crop_h, crop_w])
        else:
            offset_h = random.randint(0, image.shape[0] - crop_h)
            offset_w = random.randint(0, image.shape[1] - crop_w)
            image = image[offset_h:crop_h+offset_h, offset_w:crop_w+offset_w]

    return image

def mean_std_scale(param, image):
    if param['scale'] is not None:
        if len(param['scale']) == 3:
            for i in range(len(param['scale'])):
                image[:,:,i] = image[:,:,i] / param['scale'][i]
        elif len(param['scale']) == 1:
            image = image / param['scale']
        else:
            raise NotImplementedError

    if param['mean'] is not None:
        if len(param['mean']) == 3:
            for i in range(len(param['mean'])):
                image[:,:,i] = image[:,:,i] - param['mean'][i]
        elif len(param['mean']) == 1:
            image = image * param['mean']
        else:
            raise NotImplementedError

    if param['std'] is not None:
        if len(param['std']) == 3:
            for i in range(len(param['std'])):
                image[:,:,i] = image[:,:,i] / param['std'][i]
        elif len(param['std']) == 1:
            image = image * param['std']
        else:
            raise NotImplementedError

    return image


def hwc2chw_unsqueeze(param, image):
    framework = param['platform'].lower()
    if framework == 'tflite' or framework == 'tensorflow':
        return np.expand_dims(image, axis=0)
    else: 
        return np.expand_dims(image, axis=0).transpose(0, 3, 1, 2)


def preprocess(param):
    param['source'] = os.path.abspath(param['source'])
    source_path = param['source']
    source_path = os.path.dirname(source_path) + '/'
    param['root_folder'] = source_path

    source_file_name, ext = param['source'].split('.')
    ext = ext.lower()
    support_file = ['jpg', 'bmp', 'png', 'jpeg']

    if param['color_format'] == 'BGR':
        param['mean'] = RGB2BGR(param['mean'])
        param['std'] = RGB2BGR(param['std'])
        param['scale'] = RGB2BGR(param['scale'])

    if ext == 'txt':
        images = np.loadtxt(param['source'], dtype=str).tolist()
        if isinstance(images, list):
            np_tensor = None
            i = 0
            for image in images:
                image = get_image(param, source_path + image)
                image = mean_std_scale(param, image)
                if np_tensor is None:
                    np_tensor = hwc2chw_unsqueeze(param, image)
                else:
                    np_tensor = np.concatenate((np_tensor, hwc2chw_unsqueeze(param, image)), axis=0)
                i = i + 1
                if param['batch_size'] is not None and i >= param['batch_size']:
                    break
        elif isinstance(images, str):
            image = get_image(param, source_path + images)
            np_tensor = mean_std_scale(param, image)
    elif ext in support_file:
        image = get_image(param, param['source'])
        np_tensor = mean_std_scale(param, image)
    elif ext == 'bin':
        np_tensor = get_bin(param, param['source'])
        np_tensor = mean_std_scale(param, np_tensor)
    else:
        raise NotImplementedError('Do not support file format '+ ext)

    np_tensor = np.ascontiguousarray(np_tensor).astype(np.float32)

    framework = param['platform'].lower()
    if framework == 'tensorflow' or framework == 'tflite':
        import tensorflow as tf
        input_tensor = tf.convert_to_tensor(np_tensor)
    elif framework == 'pytorch':
        input_tensor = torch.from_numpy(np_tensor)#.to(device)
    elif framework == 'onnx':
        input_tensor = np_tensor
    else:
        raise NotImplementedError
    return input_tensor


def Convert(model_file, caffe_model_name, caffe_model_path=None, dump_level=-1, param=None):
    input_tensor = preprocess(param)

    framework = param['platform'].lower()
    if framework == 'pytorch':
        raise NotImplementedError
        PytorchConvert(model_file, input_tensor, caffe_model_name, caffe_model_path, dump_level, param)
    elif framework == 'tensorflow' or framework == 'tflite':
        TensorflowConvert(model_file, input_tensor, caffe_model_name, caffe_model_path, dump_level, param)
    elif framework == 'onnx':
        from onnx2Caffe.convertCaffe import convert as ONNXConvert
        ONNXConvert(model_file, input_tensor, caffe_model_name, caffe_model_path, dump_level, param)
    else:
        raise NotImplementedError


def args_():
    args = argparse.ArgumentParser(description = 'Pytorch/Tensorflow lite/ONNX to Caffe converter usage', epilog = '')
    args.add_argument('-platform',      type = str,     required = True, choices=['Tensorflow', 'Onnx', 'Pytorch', 'Tflite', 'tensorflow', 'ONNX', 'pytorch', 'onnx'],
            help = 'Pytorch/Tensorflow/ONNX')
    args.add_argument('-model',         type = str,     required = True,
            help = 'Orginal Model File')
    args.add_argument('-output',        type = str,     required = False,
            help = 'Caffe Model path')
    args.add_argument('-name',          type = str,     required = False,
            help = 'output model name')
    args.add_argument('-source',        type = str,     required = True,
            help = 'Specify the data source')
    args.add_argument('-batch_size',    type = int,     required = False,
            help = 'Specify batch_size')
    args.add_argument('-dtype',         type = str,     required = False,   choices=['u8', 's8', 's16'],   default = 'u8',
            help = 'Specify the Data type, only 0:u8 1:s16 2:f32')
    args.add_argument('-color_format',  type = str,     required = False,   choices=['BGR', 'RGB', 'GRAY'],
            help = 'Specify the images color format, 0:BGR 1:RGB 2:GRAY')
    args.add_argument('-scale',         type = float,   required = False,   nargs='+', default = [1, 1, 1],
            help = 'scale value')
    args.add_argument('-mean',          type = float,   required = False,   nargs='+',
            help = 'mean value')
    args.add_argument('-std',           type = float,   required = False,   nargs='+',
            help = 'std value')
    args.add_argument('-new_height',    type = int,     required = False,
            help = 'Resize images if new_height or new_width are not zero')
    args.add_argument("-new_width",     type = int,     required = False,
            help = 'Resize images if new_height or new_width are not zero')
    args.add_argument('-crop_h',        type = int,     required = False,
            help = 'Specify if we would like to randomly crop input image')
    args.add_argument('-crop_w',        type = int,     required = False,
            help = 'Specify if we would like to randomly crop input image')
    args.add_argument('-dump',          type = int,     required = False,   default = -1,   choices=[0, 1, 2],
            help = 'dump blob  1:print output.  2:print input & ouput')
    args = args.parse_args()
    return args


def main():
    args = args_()
    param = args.__dict__
    param['model'] = os.path.abspath(param['model'])
    model_file = param['model']

    if not os.path.isfile(model_file):
        print('Model File not exist!')
        return
    else:
        opt_path, opt_file_name = os.path.split(model_file)
        opt_model_name, ext = opt_file_name.split('.')

    caffe_model_name = param['name'] if param['name'] is not None else opt_model_name
    caffe_model_path = os.path.abspath(param['output']) if param['output'] is not None else opt_path
    dump_level = param['dump']
    Convert(model_file, caffe_model_name, caffe_model_path, dump_level, param)

    # Delete all argmuments
    for i in range(1, len(sys.argv)):
        sys.argv.pop()

    print('Conversion Done')


if __name__ == "__main__":
    sys.exit(main())


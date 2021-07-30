import os
import sys
import cv2
import argparse
import numpy as np


trans_dtype = {
    'u8': np.uint8,
    's8': np.int8,
    's16': np.int16,
    'f32': np.float32
}

def get_batch_size_from_model(param):
    framework = param['platform'].lower()
    if framework == 'pytorch':
        raise NotImplementedError
    elif framework == 'tensorflow' or framework == 'tflite':
        from tflite.Model import Model
        with open(param['model'], 'rb') as f:
            buf = f.read()
            model = Model.GetRootAsModel(buf, 0)
        graph = model.Subgraphs(0)
        param['batch_size'] = graph.Tensors(graph.Inputs(0)).Shape(0)
        param['input_shape'] = list(graph.Tensors(graph.Inputs(0)).ShapeAsNumpy())
        print(param['input_shape'], '=========')
    elif framework == 'onnx':
        import onnx
        model = onnx.load(param['model'])
        param['batch_size'] = model.graph.input[0].type.tensor_type.shape.dim[0].dim_value
    else:
        raise NotImplementedError


def RGB2BGR(param):
    if isinstance(param, list) and len(param) == 3:
        ret_param = []
        ret_param.append(param[2])
        ret_param.append(param[1])
        ret_param.append(param[0])
        return ret_param
    else:
        return param

def set_shape(param, np_tensor, ext):
    framework = param['platform'].lower()
    if ext == 'bin':
        param['inshape'] = [param['batch_size'], param['bin_shape'][0], param['bin_shape'][1], param['bin_shape'][2]]
        if framework == 'tflite' or framework == 'tensorflow':
            param['outshape'] = [np_tensor.shape[0], np_tensor.shape[3], np_tensor.shape[1], np_tensor.shape[2]]
        else:
            param['outshape'] = [np_tensor.shape[0], np_tensor.shape[1], np_tensor.shape[2], np_tensor.shape[3]]

def get_input_data(param, path, ext):
    if ext in ['jpg', 'bmp', 'png', 'jpeg']:
        input_data = cv2.imread(path).transpose(2, 0, 1)
        assert(input_data is not None)

        if param['color_format'] == 'RGB':
            input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
    elif ext == 'bin':
        input_data = np.fromfile(path, trans_dtype[param['dtype']])
        input_data = np.array(input_data, dtype=np.float32)

        shape = param['bin_shape']
        input_data = np.reshape(input_data, shape)
    else:
        raise NotImplementedError('Do not support file format '+ ext)

    if param['new_height'] is not None and param['new_width'] is not None:
        new_h = param['new_height']
        new_w = param['new_width']
        input_data = cv2.resize(input_data, (new_h, new_w), interpolation = cv2.INTER_CUBIC)

    if param['crop_h'] is not None and param['crop_w'] is not None:
        crop_h = param['crop_h']
        crop_w = param['crop_w']
        if input_data.shape[1] < crop_h or input_data.shape[2] < crop_w:
            raise ValueError('crop size > image size, image:', input_data.shape, 'crop:', [crop_h, crop_w])
        else:
            offset_h = (input_data.shape[1] - crop_h) // 2
            offset_w = (input_data.shape[2] - crop_w) // 2
            input_data = input_data[:,offset_h:crop_h+offset_h, offset_w:crop_w+offset_w]

    return input_data


def mean_std_scale(param, image):
    if param['scale'] is not None:
        if len(param['scale']) == 3:
            for i in range(len(param['scale'])):
                image[i,:,:] = image[i,:,:] / param['scale'][i]
        elif len(param['scale']) == 1:
            image = image / param['scale']
        else:
            raise NotImplementedError

    if param['mean'] is not None:
        if len(param['mean']) == 3:
            for i in range(len(param['mean'])):
                image[i,:,:] = image[i,:,:] - param['mean'][i]
        elif len(param['mean']) == 1:
            image = image * param['mean']
        else:
            raise NotImplementedError

    if param['std'] is not None:
        if len(param['std']) == 3:
            for i in range(len(param['std'])):
                image[i,:,:] = image[i,:,:] / param['std'][i]
        elif len(param['std']) == 1:
            image = image * param['std']
        else:
            raise NotImplementedError

    return image


def hwc2chw_unsqueeze(param, image):
    framework = param['platform'].lower()
    if framework == 'tflite' or framework == 'tensorflow':
        return np.expand_dims(image, axis=0).transpose(0, 2, 3, 1)
    else: 
        return np.expand_dims(image, axis=0)


def preprocess(param):
    param['source'] = os.path.abspath(param['source'])
    source_path = param['source']
    source_path = os.path.dirname(source_path) + '/'
    if param['root_folder'] is None:
        param['root_folder'] = source_path
    else:
        param['root_folder'] = os.path.abspath(param['root_folder']) + '/'

    if param['color_format'] == 'BGR':
        param['mean'] = RGB2BGR(param['mean'])
        param['std'] = RGB2BGR(param['std'])
        param['scale'] = RGB2BGR(param['scale'])

    input_files = np.loadtxt(param['source'], dtype=str).tolist()
    if isinstance(input_files, list):
        ext = input_files[0].split('.')[-1].lower()
        np_tensor = None
        i = 0
        for input_file in input_files:
            input_data = get_input_data(param, source_path + input_file, ext)
            input_data = mean_std_scale(param, input_data)
            if param['dump'] >= 0:
                param['input_file'] = param.get('input_file', input_file.split('/')[-1])
            if np_tensor is None:
                np_tensor = hwc2chw_unsqueeze(param, input_data)
            else:
                np_tensor = np.concatenate((np_tensor, hwc2chw_unsqueeze(param, input_data)), axis=0)
            i = i + 1
            if param['batch_size'] is not None and i >= param['batch_size']:
                break
    elif isinstance(input_files, str):
        ext = input_files.split('.')[-1].lower()
        input_data = get_input_data(param, source_path + input_files, ext)
        input_data = mean_std_scale(param, input_data)
        np_tensor = hwc2chw_unsqueeze(param, input_data)
        param['input_file'] = param.get('input_file', input_files.split('/')[-1])


    set_shape(param, np_tensor, ext)

    np_tensor = np.ascontiguousarray(np_tensor).astype(np.float32)

    framework = param['platform'].lower()
    if framework == 'tensorflow' or framework == 'tflite':
        import tensorflow as tf
        input_tensor = tf.convert_to_tensor(np_tensor)
    elif framework == 'pytorch':
        import torch
        input_tensor = torch.from_numpy(np_tensor)#.to(device)
    elif framework == 'onnx':
        input_tensor = np_tensor
    else:
        raise NotImplementedError
    return input_tensor

def dummy_input(param):
    return np.random.rand(param['input_shape'][0], param['input_shape'][1], param['input_shape'][2], param['input_shape'][3])

def Convert(model_file, caffe_model_name, caffe_model_path=None, dump_level=-1, param=None):
    get_batch_size_from_model(param)
    if param['dummy'] is True:
        input_tensor = dummy_input(param)
    else:
        input_tensor = preprocess(param)

    framework = param['platform'].lower()
    if framework == 'pytorch':
        raise NotImplementedError
        PytorchConvert(model_file, input_tensor, caffe_model_name, caffe_model_path, dump_level, param)
    elif framework == 'tensorflow' or framework == 'tflite':
        from tflite2caffe.convert import convert as TensorflowConvert
        TensorflowConvert(model_file, input_tensor, caffe_model_name, caffe_model_path, dump_level, param)
    elif framework == 'onnx':
        from onnx2caffe.convert import convert as OnnxConvert
        OnnxConvert(model_file, input_tensor, caffe_model_name, caffe_model_path, dump_level, param)
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
    args.add_argument('-root_folder',   type = str,     required = False,
            help = 'Specify the root folder')
    args.add_argument('-batch_size',    type = int,     required = False,
            help = 'Specify batch_size')
    args.add_argument('-dtype',         type = str,     required = False,   choices=['u8', 's16', 'f32'],   default = 'u8',
            help = 'Specify the Data type, 0:u8 1:s16 2:f32')
    args.add_argument('-bin_shape',     type = int,     required = False,   nargs='+',
            help = 'Specify the Data shape when input data is bin file')
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
    args.add_argument('-dump',          type = int,     required = False,   default = -1,   choices=[0, 1, 2, 3],
            help = 'dump blob  1:print output.  2:print input & ouput')
    args.add_argument('-dummy',        type = bool,     required = False,
            help = 'dummy input data')
    args.add_argument('-compare',       type = int,     required = False,   default = -1,   choices=[0, 1],
            help = '')
    args = args.parse_args()
    return args


def main():
    args = args_()
    param = args.__dict__
    param['model'] = os.path.abspath(param['model'])
    model_file = param['model']
    os.environ['GLOG_minloglevel'] = '0'

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


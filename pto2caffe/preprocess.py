import os
import sys
import cv2
import numpy as np


def load_file2tensor(path, param):
    '''
    Load image or raw data as tensor[C, H, W]
    '''
    ext = os.path.basename(path).split('.')[-1].lower()
    if ext in ['bin']:
        tensor = np.fromfile(path, param['dtype'])
        tensor = np.array(tensor, dtype=np.float32)
        tensor = np.reshape(tensor , param['bin_shape'])
        param['source_shape'] = [1] + list(tensor.shape)
    elif ext in ['jpg', 'bmp', 'png', 'jpeg']:
        tensor = cv2.imread(path)
        assert(tensor is not None), 'Error: Input file is None  ' + path
        if param['color_format'] == 'RGB':
            tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)
        tensor = tensor.transpose(2, 0, 1).astype(np.float32) # HWC->CHW
    else:
        errorMsg = 'Do not support input file format: ' + ext
        sys.exit(errorMsg)

    return tensor


def get_one_file(folder_path):
    folder = os.walk(folder_path)
    for path, dir_list, file_list in folder:
        for file_name in file_list:
            if file_name.split('.')[-1].lower() in ['jpg', 'bmp', 'png', 'jpeg', 'bin']:
                return path+'/'+file_name


def get_input_tensor(param, input_shape, dtype, quantization_parameter=None):
    root_path = param.get('root_folder', None)

    if root_path is not None and os.path.isfile(root_path):
        tensor = load_file2tensor(root_path, param)

    elif root_path is not None and os.path.isdir(root_path):
        param['input_file'] = get_one_file(root_path)
        tensor = load_file2tensor(param['input_file'], param)

    elif np.issubdtype(dtype, np.integer) and quantization_parameter is not None:
        maxval = 1. if isinstance(quantization_parameter['maxval'], int) else quantization_parameter['maxval']
        minval = -1. if isinstance(quantization_parameter['minval'], int) else quantization_parameter['minval']
        tensor = np.random.uniform(low=minval, high=maxval, size=input_shape).astype(np.float32)

    elif np.issubdtype(dtype, np.integer) and quantization_parameter is None:
        tensor = np.random.randint(low=np.iinfo(dtype).min, high=np.iinfo(dtype).max, size=input_shape, dtype=dtype)
    elif np.issubdtype(dtype, np.floating) and quantization_parameter is None:
        tensor = np.random.uniform(size=input_shape).astype(dtype)
    else:
        errorMsg = 'Can\'t support model data type: ' + str(dtype)
        sys.exit(errorMsg)

    return preprocess(tensor, param)


def mean(tensor, means):
    if len(means) == tensor.shape[1]:
        for i, mean in enumerate(means):
            tensor[:, i, :, :] = tensor[:, i, :, :] - mean
    else:
        errorMsg = 'mean length not equal tensor channel (' + str(len(means)) + ' != ' + str(tensor.shape[0]) + ')'
        sys.exit(errorMsg)

    return tensor


def crop(tensor, crop_h, crop_w):
    if tensor.shape[-2] < crop_h or tensor.shape[-1] < crop_w:
        errorMsg = 'crop size > tensor size, tensor:', tensor.shape, 'crop:', [crop_h, crop_w]
        sys.exit(errorMsg)
    else:
        offset_h = (tensor.shape[-2] - crop_h) // 2
        offset_w = (tensor.shape[-1] - crop_w) // 2
        tensor = tensor[:, offset_h:crop_h+offset_h, offset_w:crop_w+offset_w]

    return tensor


def std(tensor, stds):
    if len(stds) == tensor.shape[1]:
        for i, std in enumerate(stds):
            tensor[:, i, :, :] = tensor[:, i, :, :] / std
    else:
        errorMsg = 'std length not equal tensor channel (' + str(len(stds)) + ' != ' + str(tensor.shape[0]) + ')'
        sys.exit(errorMsg)

    return tensor


def scale(tensor, scales):
    if len(scales) == tensor.shape[1]:
        for i, scale in enumerate(scales):
            tensor[: i, :, :] = tensor[:, i, :, :] / scale
    elif len(scales) == 1:
        tensor = tensor / scales[0]
    else:
        errorMsg = 'scale length not equal tensor channel (' + str(len(scales)) + ' != ' + str(tensor.shape[0]) + ')'
        sys.exit(errorMsg)

    return tensor


def preprocess(tensor, param):
    '''
        Input tensor should be 3 dimensions and has [C, H, W] layout
    '''
    if param['scale'] is not None:
        tensor = scale(tensor, param['scale'])

    if param['mean'] is not None:
        tensor = mean(tensor, param['mean'])

    if param['std'] is not None:
        tensor = std(tensor, param['std'])

    if param['crop_h'] is not None and param['crop_w'] is not None and param.get('root_folder', None) is not None:
        tensor = crop(tensor, param['crop_h'], param['crop_w'])

    if param.get('savetensor', 0) == 1:
        model_ext = param['model'].split('.')[-1]
        bin_file = param['model'][:-len(model_ext)] + 'bin'
        tensor.tofile(bin_file)

    if param.get('root_folder', None) is None:
        return tensor
    else:
        return np.expand_dims(tensor, axis=0).astype(np.float32) # Unsqueeze CHW->NCHW

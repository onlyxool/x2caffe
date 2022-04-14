import os
import sys
import cv2
import numpy as np


numpy_dtype = { 
    'u8': np.uint8,
    's8': np.int8,
    's16': np.int16,
    'f32': np.float32
}


def load_file2tensor(path, param):
    '''
    Load image or raw data as tensor[C, H, W]
    '''
    ext = os.path.basename(path).split('.')[-1].lower()
    if ext in ['bin']:
        tensor = np.fromfile(path, numpy_dtype[param['dtype']])
        tensor = np.array(tensor, dtype=np.float32)
        tensor = np.reshape(tensor , param['bin_shape'])
    elif ext in ['jpg', 'bmp', 'png', 'jpeg']:
        tensor = cv2.imread(path).transpose(2, 0, 1).astype(np.float32) # HWC->CHW
        assert(tensor is not None), 'Error: Input file is None  ' + path
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


def get_input_tensor(path, param):
    if os.path.isfile(path):
        return load_file2tensor(path, param)
    elif os.path.isdir(path):
        param['input_file'] = get_one_file(path)
        return load_file2tensor(param['input_file'], param)
    else:
        return None

    return tensor


def mean(tensor, means):
    if len(means) == tensor.shape[0]:
        for i, mean in enumerate(means):
            tensor[i,:,:] = tensor[i,:,:] - mean
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
    if len(stds) == tensor.shape[0]:
        for i, std in enumerate(stds):
            tensor[i,:,:] = tensor[i,:,:] / std
    else:
        errorMsg = 'std length not equal tensor channel (' + str(len(stds)) + ' != ' + str(tensor.shape[0]) + ')'
        sys.exit(errorMsg)

    return tensor


def scale(tensor, scales):
    if len(scales) == tensor.shape[0]:
        for i, scale in enumerate(scales):
            tensor[i,:,:] = tensor[i,:,:] / scale
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

    if param['crop_h'] is not None and param['crop_w'] is not None:
        tensor = crop(tensor, param['crop_h'], param['crop_w'])

    return np.expand_dims(tensor, axis=0).astype(np.float32) # Unsqueeze CHW->NCHW

import sys
import numpy as np


def gen_input_tensor(param, input_shape, dtype, quantization_parameter=None):

    if np.issubdtype(dtype, np.integer) and quantization_parameter is not None:
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
    elif len(means) == tensor.shape[3]:
        for i, mean in enumerate(means):
            tensor[:, :, :, i] = tensor[:, :, :, i] - mean
    elif len(means) == 1:
        tensor = tensor - means[0]
    else:
        sys.exit('mean length not equal tensor channel.')

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
    elif len(stds) == tensor.shape[3]:
        for i, std in enumerate(stds):
            tensor[:, :, :, i] = tensor[:, :, :, i] / std
    elif len(stds) == 1:
        tensor = tensor / stds[0]
    else:
        sys.exit('std length not equal tensor channel.')

    return tensor


def scale(tensor, scales):
    if len(scales) == tensor.shape[1]:
        for i, scale in enumerate(scales):
            tensor[: i, :, :] = tensor[:, i, :, :] / scale
    elif len(scales) == tensor.shape[3]:
        for i, scale in enumerate(scales):
            tensor[: :, :, i] = tensor[:, :, :, i] / scale
    elif len(scales) == 1:
        tensor = tensor / scales[0]
    else:
        sys.exit('scale length not equal tensor channel.')

    return tensor


def preprocess(tensor, param):
    if param['scale'] is not None:
        tensor = scale(tensor, param['scale'])

    if param['mean'] is not None:
        tensor = mean(tensor, param['mean'])

    if param['std'] is not None:
        tensor = std(tensor, param['std'])

    if param['crop_h'] is not None and param['crop_w'] is not None and param.get('root_folder', None) is not None:
        tensor = crop(tensor, param['crop_h'], param['crop_w'])

    if param.get('root_folder', None) is None:
        return tensor
    else:
        return np.expand_dims(tensor, axis=0).astype(np.float32) # Unsqueeze CHW->NCHW

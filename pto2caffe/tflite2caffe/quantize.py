import sys
import numpy as np


def checkQuantilize(tensor, scales, zero_points, dtype, quantized_dimension):
    if quantized_dimension != 0:
        raise NotImplementedError

    if dtype == np.uint8:
        min = 0
        max = 255
    elif dtype == np.int8:
        raise NotImplementedError
    elif dtype == np.float16:
        raise NotImplementedError
    else:
        raise NotImplementedError

    for i, tensor_slice in enumerate(tensor[0:tensor.shape[quantized_dimension],...]):
        if tensor_slice.max() / scales[i] + zero_points[i] > max:
            errorMsg = 'Error: Input data Max value: ' + str(tensor.max()) + ' > ' + str((255 - zero_points[i]) * scales[i]) + '\n'
            sys.exit(errorMsg)

        if tensor_slice.min() / scales[i] + zero_points[i] < min:
            errorMsg = 'Error: Input data Min Value: ' + str(tensor.min()) + ' < ' + str((0 - zero_points[i]) * scales[i]) + '\n'
            sys.exit(errorMsg)

    return True


def isQuantilize(scales_len, zero_points_len):
    if scales_len == 0 and zero_points_len == 0:
        return False
    else:
        return True


def Quantize(tensor, scales, zero_points, quantized_dimension, dtype):
    if quantized_dimension == 0:
        tensor_scaled = np.divide(tensor, scales)
        tensor_quant = np.add(tensor_scaled, zero_points)
#        tensor_quant = tensor/scales + zero_points
    else:
        raise NotImplementedError

    return tensor_quant.astype(dtype)


def Dequantize(tensor, scales, zero_points, quantized_dimension, dtype):
    if quantized_dimension != 0:
        raise NotImplementedError

    if len(scales) == tensor.shape[quantized_dimension] and len(zero_points) == tensor.shape[quantized_dimension]:
        tensor_fp32_slice = list()
        tensor_int32 = tensor.astype('int32')
        if quantized_dimension == 0:
            for i in range(tensor.shape[quantized_dimension]):
                tensor_fp32_slice.append((tensor_int32[i:i+1,...] - zero_points[i]) * scales[i])
        else:
            raise NotImplementedError

        tensor_fp32 = tensor_fp32_slice[0]
        del tensor_fp32_slice[0]
        for t in tensor_fp32_slice:
            tensor_fp32 = np.concatenate((tensor_fp32, t), axis=quantized_dimension)
    elif len(scales) == 1 and len(zero_points) == 1:
        tensor_int32 = tensor.astype('int32')
        tensor_shiftted = np.subtract(tensor_int32, zero_points)
        tensor_fp32 = np.multiply(tensor_shiftted.astype(dtype), scales)
#       tensor_fp32 = (tensor_int32 - zero_points) * scales
    else:
        raise NotImplementedError

    return tensor_fp32


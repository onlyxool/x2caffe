from math import ceil

def computePad(layer_type, operator_attrs, input_shape, output_shape, kernel_size, strides, legacy_pad):

    auto_pad_mode = operator_attrs.get('auto_pad', b'NOTSET').decode('utf-8')
    if auto_pad_mode != 'NOTSET' and auto_pad_mode != 'VALID':
        # Compute Pad
        if layer_type == 'Convolution':
            pad_h = output_shape[2] - ((input_shape[2] - kernel_size[0]) / strides[0] + 1) # output_shape - compute_output_shape
            pad_w = output_shape[3] - ((input_shape[3] - kernel_size[1]) / strides[1] + 1)
        elif layer_type == 'Deconvolution':
            pad_h = output_shape[2] - (stride_h * (input_shape[2] - 1) + kernel_size[0])
            pad_w = output_shape[3] - (stride_w * (input_shape[3] - 1) + kernel_size[1])
        elif layer_type == 'Pooling':
            pad_h = (output_shape[2] - 1) * strides[0] + ((kernel_size[0] - 1)+1) - input_shape[2]
            pad_w = (output_shape[3] - 1) * strides[1] + ((kernel_size[1] - 1)+1) - input_shape[3]


        if layer_type == 'Convolution' or layer_type == 'Deconvolution':
            pad_t = pad_b = ceil(pad_h / 2)
            pad_l = pad_r = ceil(pad_w / 2)
        elif layer_type == 'Pooling':
            pad_t = pad_b = int(pad_h / 2)
            pad_l = pad_r = int(pad_w / 2)


        if (pad_h % 2) != 0:
            if auto_pad_mode == 'SAME_UPPER':
                pad_b += 1
            elif auto_pad_mode == 'SAME_LOWER':
                pad_t += 1

        if (pad_w % 2) != 0:
            if auto_pad_mode == 'SAME_UPPER':
                pad_r += 1
            elif auto_pad_mode == 'SAME_LOWER':
                pad_l += 1
    else:
        # Attribute Pad
        pad_t,pad_l,pad_b,pad_r = operator_attrs.get('pads', [0,0,0,0])

    # Legacy Pad
    pad_l += legacy_pad['left']
    pad_r += legacy_pad['right']
    pad_t += legacy_pad['top']
    pad_b += legacy_pad['bottom']


    if pad_l == pad_r and pad_t == pad_b:
        return {'pad_w': pad_l, 'pad_h': pad_t}
    else:
        return {'pad_l': pad_l, 'pad_r': pad_r, 'pad_t': pad_t, 'pad_b': pad_b}

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Convolution(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.convolution_param = dict()
        assert(self.operator_code == 'Conv')
        self.setInited()


    def parse(self):
        self.layer_type = 'Convolution'
        super().__parse__()

        # Weight
        self.weight = self.inputs_buf[1]

        # Bias
        self.bias = self.inputs_buf[2] if len(self.inputs_buf) == 3 else None

        # Attributes
        self.convolution_param['num_output'] = self.weight.shape[0]

        self.convolution_param['stride_h'] = stride_h = self.attrs.get('strides', [1, 1])[0]
        self.convolution_param['stride_w'] = stride_w = self.attrs.get('strides', [1, 1])[1]

        self.convolution_param['dilation'] = self.attrs.get('dilations', [1, 1])
        self.convolution_param['group'] = self.attrs.get('group', 1)
        self.convolution_param['kernel_size'] = self.attrs['kernel_shape']
        self.convolution_param['bias_term'] = True if self.bias is not None else False

        # Padding
        pad_t,pad_l,pad_b,pad_r = self.attrs.get('pads', [0,0,0,0])

        auto_pad_mode = self.attrs.get('auto_pad', b'NOTSET').decode('utf-8')
        if auto_pad_mode != 'NOTSET' and auto_pad_mode != 'VALID':
            output_h = (self.inputs_shape[0][2]-self.inputs_shape[1][2]) / stride_h + 1
            output_w = (self.inputs_shape[0][3]-self.inputs_shape[1][3]) / stride_w + 1

            pad_h = self.outputs_shape[0][2] - output_h
            pad_w = self.outputs_shape[0][3] - output_w

            from math import ceil
            pad_t = pad_b = ceil(pad_h / 2)
            if (pad_h % 2) != 0:
                if auto_pad_mode == 'SAME_UPPER':
                    pad_b += 1
                elif auto_pad_mode == 'SAME_LOWER':
                    pad_t += 1

            pad_l = pad_r = ceil(pad_w / 2)
            if (pad_w % 2) != 0:
                if auto_pad_mode == 'SAME_UPPER':
                    pad_r += 1
                elif auto_pad_mode == 'SAME_LOWER':
                    pad_l += 1

        for legacy in self.model.legacys:
            if legacy.outputs[0] == self.inputs[0] and legacy.operator == 'Pad':
                legacy_pad = legacy.pad
                pad_l += legacy.pad['left']
                pad_r += legacy.pad['right']
                pad_t += legacy.pad['top']
                pad_b += legacy.pad['bottom']
                self.inputs[0] = legacy.inputs[0]
                self.inputs_shape[0] = legacy.inputs_shape[0]

        if pad_l == pad_r and pad_t == pad_b:
            self.convolution_param['pad_w'] = pad_l
            self.convolution_param['pad_h'] = pad_t
        else:
            self.convolution_param['pad_l'] = pad_l
            self.convolution_param['pad_r'] = pad_r
            self.convolution_param['pad_t'] = pad_t
            self.convolution_param['pad_b'] = pad_b

        self.attrs = self.convolution_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, convolution_param=self.convolution_param)

        self.setConverted()

        return [layer]

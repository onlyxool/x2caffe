from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Pooling(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code in ('MaxPool', 'AveragePool', 'GlobalAveragePool'))
        self.setInited()


    def parse(self):
        self.layer_type = 'Pooling'
        super().__parse__()

        # Pooling
        self.pooling_param = dict()
        if self.operator_code == 'MaxPool':
            self.pooling_param['pool'] = 0
            self.pooling_param['kernel_h'] = kernel_h = self.attrs['kernel_shape'][0]
            self.pooling_param['kernel_w'] = kernel_w = self.attrs['kernel_shape'][1]
        elif self.operator_code == 'AveragePool':
            self.pooling_param['pool'] = 1
            self.pooling_param['kernel_h'] = kernel_h = self.attrs['kernel_shape'][0]
            self.pooling_param['kernel_w'] = kernel_w = self.attrs['kernel_shape'][1]
        elif self.operator_code == 'GlobalAveragePool':
            self.pooling_param['pool'] = 1
            self.pooling_param['global_pooling'] = True
        else:
            raise NotImplementedError(self.operator_code)

        if 'dilations' in self.attrs and self.attrs['dilations'] != [1, 1]:
            raise NotImplementedError('Caffe Pooling don\'t support dilation')

        # Attributes
        strides = self.attrs.get('strides', [1, 1])
        self.pooling_param['stride_h'] = strides[0]
        self.pooling_param['stride_w'] = strides[1]
        self.pooling_param['ceil_mode'] = True if self.attrs.get('ceil_mode', False) else False

        # Padding
        pad_t,pad_l,pad_b,pad_r = self.attrs.get('pads', [0,0,0,0])

        auto_pad_mode = self.attrs.get('auto_pad', b'NOTSET').decode('utf-8')
        if auto_pad_mode != 'NOTSET' and auto_pad_mode != 'VALID':
            pad_h = (self.outputs_shape[0][2] - 1) * strides[0] + ((kernel_h - 1)+1) - self.inputs_shape[0][2]
            pad_w = (self.outputs_shape[0][3] - 1) * strides[1] + ((kernel_w - 1)+1) - self.inputs_shape[0][3]

            pad_t = pad_b = int(pad_h / 2)
            if (pad_h % 2) != 0:
                if auto_pad_mode == 'SAME_UPPER':
                    pad_b += 1
                elif auto_pad_mode == 'SAME_LOWER':
                    pad_t += 1

            pad_l = pad_r = int(pad_w / 2)
            if (pad_w % 2) != 0:
                if auto_pad_mode == 'SAME_UPPER':
                    pad_r += 1
                elif auto_pad_mode == 'SAME_LOWER':
                    pad_l += 1

        for legacy in self.model.legacys:
            if legacy.outputs[0] == self.inputs[0] and legacy.operator_code == 'Pad':
                legacy_pad = legacy.pad
                pad_l += legacy.pad['left']
                pad_r += legacy.pad['right']
                pad_t += legacy.pad['top']
                pad_b += legacy.pad['bottom']
                self.inputs[0] = legacy.inputs[0]
                self.inputs_shape[0] = legacy.inputs_shape[0]

        if pad_l == pad_r and pad_t == pad_b:
            self.pooling_param['pad_w'] = pad_l
            self.pooling_param['pad_h'] = pad_t
        else:
            self.pooling_param['pad_l'] = pad_l
            self.pooling_param['pad_r'] = pad_r
            self.pooling_param['pad_t'] = pad_t
            self.pooling_param['pad_b'] = pad_b

        self.attrs = self.pooling_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param)

        self.setConverted()

        return [layer]

from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator


class Pooling(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        assert(self.operator_code in ('nn.AvgPool2d', 'nn.MaxPool2d'))
        self.setInited()


    def parse(self):
        self.type = 'Pooling'
        super().__parse__()

        self.pooling_param = dict()
        if self.operator_code == 'nn.MaxPool2d':
            self.pooling_param['pool'] = 0 
            self.pooling_param['kernel_h'] = kernel_h = self.attrs['kernel_size'][0]
            self.pooling_param['kernel_w'] = kernel_w = self.attrs['kernel_size'][1]
        elif self.operator_code == 'nn.AvgPool2d':
            self.pooling_param['pool'] = 1
            self.pooling_param['kernel_h'] = kernel_h = self.attrs['kernel_size'][0]
            self.pooling_param['kernel_w'] = kernel_w = self.attrs['kernel_size'][1]

        if 'dilations' in self.attrs and self.attrs['dilations'] != [1, 1]:
            errorMsg = 'Caffe Pooling don\'t support dilation' + self.attrs['dilations']
            raise NotImplementedError(errorMsg)

        self.pooling_param['stride_h'] = self.attrs.get('stride', [1, 1])[0]
        self.pooling_param['stride_w'] = self.attrs.get('stride', [1, 1])[1]
        self.pooling_param['pad_h'] = self.attrs.get('padding', [0,0])[0]
        self.pooling_param['pad_w'] = self.attrs.get('padding', [0,0])[1]
        self.pooling_param['ceil_mode'] = self.attrs.get('ceil_mode', True)

        self.attrs = self.pooling_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param)

        self.setConverted()

        return [layer]

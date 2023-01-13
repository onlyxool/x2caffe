from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class ReLUX(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Clip')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[2] is not None:
            self.type = 'ReLUX'

            self.relux_param = dict()
            if 'max' in self.attrs and 'min' in self.attrs:
                self.relux_param['x'] = self.attrs['max']
            else:
                self.relux_param['x'] = self.inputs_buf[2]
            self.attrs = self.relux_param
        else:
            self.type = 'ReLU'

            self.relu_param = dict()
            self.relu_param['negative_slope'] = self.attrs.get('alpha', 0)
            self.attrs = self.relu_param

        self.inputs = self.inputs[:1]

        self.setParsed()


    def convert(self):
        if self.type == 'ReLUX':
            layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, relux_param=self.relux_param)
        elif self.type == 'ReLU':
            layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, relu_param=self.relu_param)

        self.setConverted()

        return [layer]

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class ReLUX(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Clip')
        self.setInited()


    def parse(self):
        super().__parse__()

        self.type = 'ReLUX'
        self.relux_param = dict()

        if self.model.opset[0] >= 11:
            self.relux_param['x'] = self.inputs_buf[2] if self.inputs_buf[2] is not None else 3.402823e+38
        else:
            self.relux_param['x'] = self.attrs['max'] if 'max' in self.attrs else 3.402823e+38
            if 'min' in self.attrs and self.attrs['min'] != 0:
                print('Warning: Clip Min value is not 0')

        self.attrs = self.relux_param

        self.inputs = self.inputs[:1]

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, relux_param=self.relux_param)

        self.setConverted()

        return [layer]

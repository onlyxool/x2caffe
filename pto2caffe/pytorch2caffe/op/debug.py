from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator


class Debug(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        assert(self.operator_code == '')
        self.setInited()


    def parse(self):
        self.layer_type = 'Debug'
        super().__parse__()

        print('Debug---------')
        print(self.name, self.operator_code)
        print('input', self.inputs)
        print('input_shape', self.inputs_shape)
        print('input_buf', self.inputs_buf)
        print('output', self.outputs, 'output_shape',self.outputs_shape)
        # Attributes
        self.debug_param = dict()
        print(self.attrs)
        print('Debug')

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, debug_param=self.debug_param)

        self.setConverted()

        return [layer]

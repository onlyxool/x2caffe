from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Debug(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.setInited()


    def parse(self):
        self.type = 'Debug'
        super().__parse__()
        print('====================')
        print('Op:', self.name, self.operator_code)
        print('Input:', self.inputs, self.inputs_shape)
        print('Output:', self.outputs, self.outputs_shape)
        print('Attrs:', self.attrs)
        print('Buf', self.inputs_buf)
        print('--------------\n')
        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, debug_param=self.debug_param)

        self.setConverted()

        return [layer]

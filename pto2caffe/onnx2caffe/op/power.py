from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Pow(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Pow')
        self.setInited()


    def parse(self):
        self.type = 'Power'
        super().__parse__()

        # Attributes
        self.power_param = dict()
        self.power_param['power'] = self.inputs_buf[1]
        self.power_param['scale'] = 1
        self.power_param['shift'] = 0

        self.attrs = self.power_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, power_param=self.power_param)

        self.setConverted()

        return [layer]

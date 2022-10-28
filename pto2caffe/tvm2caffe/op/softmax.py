from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator


class Softmax(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'nn.softmax')
        self.setInited()


    def parse(self):
        self.type = 'Softmax'
        super().__parse__()

        if self.inputs_buf[0] is not None:
            import numpy as np
            max = np.max(self.inputs_buf[0], axis=1, keepdims=True) #returns max of each row and keeps same dims
            e_x = np.exp(self.inputs_buf[0] - max) #subtracts each row with its max value
            sum = np.sum(e_x, axis=1, keepdims=True) #returns sum of each row and keeps same dims
            f_x = e_x / sum
            self.saveConstant(self.outputs[0], f_x)
        else:
            self.softmax_param = dict()
            self.softmax_param['axis'] = self.attrs.get('axis', 1)
            self.attrs = self.softmax_param

            self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, softmax_param=self.softmax_param)

        self.setConverted()

        return [layer]

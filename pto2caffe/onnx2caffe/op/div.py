from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Div(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.setInited()


    def parse(self):
        self.layer_type = 'Scale'
        super().__parse__()

        # Attributes
        if self.inputs_buf[1] is not None:
            self.scale_param = dict()
            self.scale_param['bias_term'] = False
            for i in range(len(self.inputs_shape[0])):
                if self.inputs_shape[1] == [] or self.inputs_shape[1] == ():
                    self.scale_param['axis'] = 0
                    break
                if self.inputs_shape[0][i] == self.inputs_shape[1][0]:
                    self.scale_param['axis'] = i
                    break
            self.attrs = self.scale_param

            # Weight
            self.weight = 1/self.inputs_buf[1]

            # Bias
            self.bias = None
        else:
            raise NotImplementedError(self.operator)

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)

        self.setConverted()

        return [layer]
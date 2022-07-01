from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class RealDiv(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'RealDiv')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            self.model.constant[self.outputs[0]] = self.inputs_buf[0] / self.inputs_buf[1]
        elif self.inputs_buf[1] is not None or self.inputs_buf[0] is not None:
            self.layer_type = 'Scale'

            if self.inputs_buf[0] is not None:
                self.inputs.reverse()
                self.inputs_shape.reverse()
                self.inputs_buf.reverse()

            self.scale_param = dict()
            self.scale_param['bias_term'] = False
            self.scale_param['axis'] = 0

            self.weight = 1 / self.inputs_buf[1]
            self.bias = None

            self.attrs = self.scale_param
            self.setParsed()
        else:
            raise NotImplementedError


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)

        self.setConverted()

        return [layer]

import tflite

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator


class Slice(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        assert(self.operator_code == 'SLICE')
        assert(self.op.InputsLength() == 3)
        assert(self.op.OutputsLength() == 1)
        self.setInited()


    def parse(self):
        self.parseInputOutput()

        if self.inputs_shape[0] == self.outputs_shape[0]:
            self.byPassOperator()
        else:
            raise NotImplementedError
            self.layer_type = 'Slice'
            self.slice_param = dict()
            self.slice_param['axis'] = 0
            self.slice_param['slice_point'] = slice_points

            self.attrs = self.slice_param

            self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, slice_param=self.slice_param)

        self.setConverted()

        return [layer]


from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator

from util import dim_map_nhwc2nchw


class Concat(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'ConcatV2')
        self.setInited()


    def parse(self):
        super().__parse__()

        for index, input in enumerate(self.inputs):
            if input.lower().find('axis') != -1:
                axis = dim_map_nhwc2nchw[int(self.inputs_buf[index])] if self.layout == 'NHWC' else int(self.inputs_buf[index])

        for input_buf in self.inputs_buf:
            if input_buf is not None:
                constant = True
            else:
                constant = False
                break

        if constant:
            import numpy as np
            self.saveConstant(self.outputs[0], np.concatenate(self.inputs_buf[:-1], axis=int(self.inputs_buf[index])))
        else:
            self.type = 'Concat'

            self.concat_param = dict()
            self.concat_param['axis'] = axis

            self.attrs = self.concat_param

            self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, concat_param=self.concat_param)

        self.setConverted()

        return [layer]

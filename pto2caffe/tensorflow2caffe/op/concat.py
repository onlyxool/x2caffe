from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator
from util import dim_map_nhwc2nchw


class Concat(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator == 'ConcatV2')
        self.setInited()


    def parse(self):
        self.layer_type = 'Concat'
        super().__parse__()

        self.concat_param = dict()
        for index, input_buf in enumerate(self.inputs_buf):
            if input_buf is not None:
                self.concat_param['axis'] = dim_map_nhwc2nchw[self.inputs_buf[index]]

        self.attrs = self.concat_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, concat_param=self.concat_param)

        self.setConverted()

        return [layer]

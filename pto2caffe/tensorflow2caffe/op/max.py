from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator
from util import handleLegacyPad, getLegacyAttrs


class Max(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Max')
        self.setInited()


    def parse(self):
        self.layer_type = 'Pooling'
        super().__parse__()

        reduction_indices = self.inputs_buf[1]
        if reduction_indices is not None:
            raise NotImplementedError

        # Attribute
        self.pooling_param = dict()
        self.pooling_param['pool'] = 0
        self.pooling_param['kernel_h'] = self.inputs_shape[0][2]
        self.pooling_param['kernel_w'] = self.inputs_shape[0][3]
        self.pooling_param['stride'] = 1
        self.pooling_param['ceil_mode'] = False

        self.attrs = self.pooling_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param)

        self.setConverted()

        return [layer]

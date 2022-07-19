from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator
from util import dim_map_nhwc2nchw


class Pack(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Pack')
        self.setInited()


    def parse(self):
        self.layer_type = 'Concat'
        super().__parse__()

        self.reshapes = list()
        for index, input in enumerate(self.inputs):
            self.reshapes.append('Pack_' + self.op.name + '_split' + str(index))
        self.reshape_param = dict(shape=dict(dim=[1]+self.inputs_shape[index]))

        self.concat_param = dict()
        self.concat_param['axis'] = dim_map_nhwc2nchw[self.attrs['axis']]

        self.attrs = self.concat_param

        self.setParsed()


    def convert(self):
        layers = list()
        for index, reshape_name in enumerate(self.reshapes):
            layers.append(caffe_layer('Reshape', reshape_name, [self.inputs[index]], [None], [reshape_name], reshape_param=self.reshape_param))

        layers.append(caffe_layer(self.type, self.name, self.reshapes, self.inputs_buf, self.outputs, concat_param=self.concat_param))

        self.setConverted()

        return layers

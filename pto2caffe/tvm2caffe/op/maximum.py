import numpy as np

from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator


class Maximum(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'maximum')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is None and self.inputs_buf[1] is None:
            self.type = 'Eltwise'
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 2
            self.attrs = self.eltwise_param
            self.setParsed()
        elif self.inputs_buf[1] is not None:
            if np.count_nonzero(self.inputs_buf[1]) > 0:
                self.type = 'DummyData+Eltwise'
                self.dummy_data_param = dict()
                self.dummy_data_param['data_filler'] = dict(type='constant', value=self.inputs_buf[1])
                self.dummy_data_param['shape'] = dict(dim=self.inputs_shape[0])

                self.inter_blob = 'preDummy_split' + str(self.index)

                self.eltwise_param = dict()
                self.eltwise_param['operation'] = 2

                self.attrs = self.eltwise_param
                self.setParsed()
            else:
                self.type = 'ReLU'
                self.relu_param = dict()
                self.relu_param['negative_slope'] = 0

                self.attrs = self.relu_param
                self.setParsed()


    def convert(self):
        layers = list()
        if self.type == 'Eltwise':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param))
        elif self.type == 'DummyData+Eltwise':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], [self.inputs[1]], [self.inputs_buf[1]], [self.inter_blob], dummy_data_param=self.dummy_data_param))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], [self.inputs[0], self.inter_blob], [None, None], self.outputs, eltwise_param=self.eltwise_param))
        elif self.type == 'ReLU':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, relu_param=self.relu_param))

        self.setConverted()

        return layers

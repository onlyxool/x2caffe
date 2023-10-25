from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator


class Minimum(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'minimum')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is None and self.inputs_buf[1] is None:
            self.type = 'Eltwise'

            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 4

            self.attrs = self.eltwise_param
            self.setParsed()
        elif self.inputs_buf[1] is not None and isinstance(self.inputs_buf[1], (int, float)):
            self.type = 'DummyData+Eltwise'
            self.dummy_data_param = dict()
            self.dummy_data_param['data_filler'] = dict(type='constant', value=self.inputs_buf[1])
            self.dummy_data_param['shape'] = dict(dim=self.inputs_shape[0])

            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 4

            self.attrs = self.eltwise_param
            self.setParsed()


    def convert(self):
        layers = list()

        if self.type == 'Eltwise':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param))
        elif self.type == 'DummyData+Eltwise':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], [self.inputs[1]], [self.inputs_buf[1]], self.interblob, dummy_data_param=self.dummy_data_param))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], [self.inputs[0], self.interblob[0]], [None, None], self.outputs, eltwise_param=self.eltwise_param))

        self.setConverted()

        return layers

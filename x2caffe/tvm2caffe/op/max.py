from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator

from util import dim_map_nhwc2nchw


class Max(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'max')
        self.setInited()


    def parse(self):
        super().__parse__()

        for index, axis in enumerate(self.attrs['axis']):
            self.attrs['axis'][index] = dim_map_nhwc2nchw[self.attrs['axis'][index]] if self.layout == 'NHWC' else self.attrs['axis'][index]

        if self.attrs['axis'] == [2, 3]:
            self.type = 'Pooling'

            self.pooling_param = dict()
            self.pooling_param['pool'] = 0
            self.pooling_param['stride'] = 1
            self.pooling_param['ceil_mode'] = False
            self.pooling_param['global_pooling'] = True

            if 'keepdims' not in self.attrs.keys() or not self.attrs['keepdims']:
                self.type = 'Pooling+Reshape'
                self.reshape = 'Max_reshape_split' + str(self.index)
                self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))

            # Padding
            pool_pad = self.model.pad.get(self.relay_inputs[0], [0, 0, 0, 0]) 
            if pool_pad[0] == pool_pad[2] and pool_pad[1] == pool_pad[3]:
                self.pooling_param['pad_h'] = pool_pad[0]
                self.pooling_param['pad_w'] = pool_pad[1]
            else:
                self.pooling_param['pad_t'] = pool_pad[0]
                self.pooling_param['pad_l'] = pool_pad[1]
                self.pooling_param['pad_b'] = pool_pad[2]
                self.pooling_param['pad_r'] = pool_pad[3]

            self.attrs = self.pooling_param
        elif len(self.attrs['axis']) == 1:
            self.type = 'ArgMax'

            self.argmax_param = dict()
            self.argmax_param['out_max_val'] = True
            self.argmax_param['top_k'] = 1 
            self.argmax_param['axis'] = self.attrs['axis'][0]

            self.attrs = self.argmax_param
        else:
            self.unSupported('Can\'t support axis == ' + str(self.attrs['axis']))
            return

        self.setParsed()


    def convert(self):
        layers = list()
        if self.type == 'ArgMax':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, argmax_param=self.argmax_param))
        elif self.type == 'Pooling':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param))
        elif self.type == 'Pooling+Reshape':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], self.inputs, self.inputs_buf, [self.reshape], pooling_param=self.pooling_param))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], [self.reshape], [None], self.outputs, reshape_param=self.reshape_param))

        self.setConverted()

        return layers

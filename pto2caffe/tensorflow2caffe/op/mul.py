import copy

from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator

from util import trim_one
from util import compute_scale_axis

class Mul(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Mul')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            self.model.constant[self.outputs[0]] = self.inputs_buf[0] * self.inputs_buf[1]
        elif self.inputs_shape[0] != self.inputs_shape[1] or self.inputs_buf[1] is not None:
            self.layer_type = 'Scale'

            if self.inputs_buf[0] is not None or self.inputs_shape[0].count(1) > self.inputs_shape[1].count(1):
                self.inputs.reverse()
                self.inputs_shape.reverse()

            org_shape = copy.deepcopy(self.inputs_shape[1])
            trim = trim_one(org_shape)
            if trim != self.inputs_shape[1]:
                self.inputs_shape[1] = trim
                if self.inputs_buf[1] is not None:
                    self.inputs_buf[1] = self.inputs_buf[1].reshape(tuple(trim))
                else:
                    self.pre = 'Reshape'

            axis = compute_scale_axis(self.inputs_shape[0], trim)

            self.scale_param = dict()
            self.scale_param['bias_term'] = False
            if axis is not None:
                self.scale_param['axis'] = axis

            self.weight = self.inputs_buf[1]
            self.bias = None

            self.attrs = self.scale_param
            self.setParsed()
        else:
            self.layer_type = 'Eltwise'

            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 0

            self.attrs = self.eltwise_param
            self.setParsed()


    def convert(self):
        layers = list()
        if self.type == 'Scale':
            if self.pre == 'Reshape':
                pre_layer = caffe_layer('Reshape', 'Reshape'+str(self.index), [self.inputs[1]], [None],
                        ['reshape'+str(self.index)], reshape_param=dict(shape=dict(dim=self.inputs_shape[1])))
                layers.append(pre_layer)
                self.inputs[1] = 'reshape' + str(self.index)
            layers.append(caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param))
        elif self.type == 'Eltwise':
            layers.append(caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param))

        self.setConverted()

        return layers

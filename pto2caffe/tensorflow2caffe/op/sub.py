import numpy as np
from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator

from util import get_layout


class Sub(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Sub')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            self.saveConstant(self.outputs[0], self.inputs_buf[0] - self.inputs_buf[1])
        elif self.inputs_buf[0] is None and self.inputs_buf[1] is None and self.inputs_shape[0] == self.inputs_shape[1]:
            self.layer_type = 'Eltwise'
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 3 # Caffe Eltwise SUB
            self.attrs = self.eltwise_param
            self.setParsed()
        elif self.inputs_buf[0] is None and self.inputs_buf[1] is not None:
            self.layer_type = 'Scale'

            # Weight
            self.weight = np.ones(self.inputs_shape[0]).astype(np.float32)

            # Bias
            if len(self.inputs_shape[1]) == 4 and self.layout == 'NHWC':
                self.bias = -self.inputs_buf[1].transpose(0, 3, 1, 2)
            elif len(self.inputs_shape[1]) == 3 and get_layout(self.op.inputs[1].shape.as_list()) == 'HWX':
                self.bias = -self.inputs_buf[1].transpose(2, 0, 1)
            else:
                self.bias = -self.inputs_buf[1]

            # Attribute
            self.scale_param = dict()
            self.scale_param['axis'] = 0
            self.scale_param['num_axes'] = len(self.weight.shape)
            self.scale_param['bias_term'] = True

            self.attrs = self.scale_param
            self.setParsed()
        elif self.inputs_buf[0] is not None and self.inputs_buf[1] is None:
            self.layer_type = 'Scale+Scale'

            self.inputs.reverse()
            self.inputs_shape.reverse()
            self.inputs_buf.reverse()

            self.scale0 = 'Scale_split_sub'+str(self.index)
            self.scale1 = 'Scale_split_neg'+str(self.index)
            self.scale0_outputs = ['Scale_split_sub'+str(self.index)]
            self.scale1_inputs = ['Scale_split_sub'+str(self.index), 'Scale_split_neg_weight']

            # Weight
            self.weight0 = np.ones(self.inputs_shape[0]).astype(np.float32)
            self.weight1 = np.ones(self.outputs_shape[0]).astype(np.float32) * -1

            # Bias
            if len(self.inputs_shape[1]) == 4:
                self.bias0 = -self.inputs_buf[1].transpose(0,3,1,2)
            elif len(self.inputs_shape[1]) == 3 and get_layout(self.inputs_shape[1]) == 'HWX':
                self.bias0 = -self.inputs_buf[1].transpose(2,0,1)
            else:
                self.bias0 = -self.inputs_buf[1]
            self.bias1 = None

            # Attribute
            self.scale_param0 = dict()
            self.scale_param0['axis'] = 0
            self.scale_param0['num_axes'] = len(self.weight0.shape)
            self.scale_param0['bias_term'] = True

            self.scale_param1 = dict()
            self.scale_param1['axis'] = 0
            self.scale_param1['num_axes'] = len(self.weight1.shape)
            self.scale_param1['bias_term'] = False

            self.attrs = self.scale_param0
            self.setParsed()
        else:
            raise NotImplementedError


    def convert(self):
        layers = list()
        if self.layer_type == 'Eltwise':
            layers.append(caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param))
        if self.layer_type == 'Scale':
            layers.append(caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param))
        if self.layer_type == 'Scale+Scale':
            layers.append(caffe_layer('Scale', self.scale0, self.inputs, self.inputs_buf, self.scale0_outputs, self.weight0, self.bias0, scale_param=self.scale_param0))
            layers.append(caffe_layer('Scale', self.scale1, self.scale1_inputs, [None, self.weight1], self.outputs, self.weight1, self.bias1, scale_param=self.scale_param1))

        self.setConverted()

        return layers

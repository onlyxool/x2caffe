from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator

from util import shape_map_nhwc2nchw


class Reshape(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Reshape')
        self.setInited()


    def parse(self):
        self.layer_type = 'Reshape'
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            import numpy as np
            self.model.constant[self.outputs[0]] = np.reshape(self.inputs_buf[0], self.inputs_buf[1])
        else:
            if None in self.outputs_shape[0]:
                import sys
                sys.exit('Error: Dynamic Model input detected, Please Use -inputs_shape overwirte input shape.')

            # Attribute
            if self.layout == 'NCHW':
                target_shape = self.outputs_shape[0]
            elif self.layout == 'NHWC':
                if self.op.inputs[0].shape.as_list() == self.inputs_shape[0] and len(self.outputs_shape[0]) != 4:
                    target_shape = self.op.outputs[0].shape.as_list()
                else:
                    target_shape = self.outputs_shape[0]

#            if self.op.outputs[0].shape.is_fully_defined():
#                target_shape = self.outputs_shape[0]
#            elif self.inputs_buf[1] is not None:
#                target_shape = shape_map_nhwc2nchw(self.inputs_buf[1].tolist()) if self.layout == 'NHWC' and len(self.inputs_shape[0]) == 4 else self.inputs_buf[1].tolist()
#            else:
#                raise NotImplementedError


            if len(self.inputs) == 2:
                self.inputs.pop()
                self.inputs_shape.pop()
                self.inputs_buf.pop()

            if self.layout == 'NHWC' and len(self.inputs_shape[0]) == 4 and len(self.inputs_shape[0]) > len(self.outputs_shape[0]):
                self.pre = 'Permute'
                self.permute = 'Reshape_' + self.op.name + '_split' + str(self.index)
                self.permute_param = dict(order=[0,2,3,1])

            self.reshape_param = dict(shape=dict(dim=target_shape))
            self.attrs = self.reshape_param

            self.setParsed()


    def convert(self):
        layers = list()
        if self.pre == 'Permute':
            layers.append(caffe_layer('Permute', self.permute, [self.inputs[0]], [None], [self.permute], permute_param=self.permute_param))
            layers.append(caffe_layer(self.type, self.name, [self.permute], self.inputs_buf, self.outputs, reshape_param=self.reshape_param))
        else:
            layers.append(caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, reshape_param=self.reshape_param))

        self.setConverted()

        return layers

import numpy as np

from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class Mul(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'mul')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            self.saveConstant(self.node.outputs[0], self.inputs_buf[0] * self.inputs_buf[1])
        elif self.inputs_buf[0] is None and self.inputs_buf[1] is None and self.inputs_shape[0] == self.inputs_shape[1]:
            self.type = 'Eltwise'
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 0 # Caffe Eltwise PROD
            self.attrs = self.eltwise_param
            self.setParsed()
        elif (self.inputs_shape[0] != self.inputs_shape[1]) or (self.inputs_buf[0] is not None or self.inputs_buf[1] is not None):
            self.type = 'Scale'

            inputs_size0 = np.multiply.reduce(self.inputs_shape[0], axis=None)
            inputs_size1 = np.multiply.reduce(self.inputs_shape[1], axis=None)

            if self.inputs_buf[0] is not None and self.inputs_buf[1] is None:
                self.inputs.reverse()
                self.inputs_shape.reverse()
                self.inputs_buf.reverse()
                self.outputs_shape[0] = self.inputs_shape[0]
                self.model.tensor_shape[self.outputs[0]] = self.outputs_shape[0]
            elif self.inputs_buf[0] is None and self.inputs_buf[1] is None and inputs_size0 < inputs_size1:
                self.inputs.reverse()
                self.inputs_shape.reverse()
                self.inputs_buf.reverse()
                self.outputs_shape[0] = self.inputs_shape[0]
                self.model.tensor_shape[self.outputs[0]] = self.outputs_shape[0]

            if self.inputs_buf[1] is not None and np.all(self.inputs_buf[1] == 1): 
                self.byPassOperator()
                return

            WeightShape = self.inputs_shape[1]
            CompatibleFlag = self.checkShapeCompatible()
            if CompatibleFlag == 'Squeeze':
                self.type = 'Reshape+Scale'
            elif not CompatibleFlag:
                self.unSupported('Inputs incompatible shapes for Caffe. ' + str(self.inputs_shape[0]) + ' x ' + str(WeightShape))
                return

            self.scale_param = dict()
            self.scale_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0]) if len(self.inputs_shape[1]) > 0 else 0
            self.scale_param['num_axes'] = len(self.inputs_shape[1])
            self.attrs = self.scale_param

            self.weight = self.inputs_buf[1]
            self.setParsed()


    def convert(self):
        if self.type == 'Eltwise':
            layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)
        elif self.type == 'Scale':
            layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, scale_param=self.scale_param)

        self.setConverted()

        return [layer]

import numpy as np

from util import trim_one
from util import compute_scale_axis
from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


def switch(items:list):
    temp = items[0]
    items[0] = items[1]
    items[1] = temp


class Add(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Add')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is None and self.inputs_buf[1] is None:
            if self.inputs_shape[0] == self.inputs_shape[1]:
                # Eltwise Layer
                self.layer_type = 'Eltwise'

                # Attributes
                self.eltwise_param = dict()
                self.eltwise_param['operation'] = 1 # Caffe Eltwise SUM
                self.attrs = self.eltwise_param
            else:
                self.layer_type = 'Bias'
                self.bias_param = dict()

                if self.inputs_shape[0].count(1) > self.inputs_shape[1].count(1):
                    switch(self.inputs)
                    switch(self.inputs_shape)

                self.inputs_shape[1] = trim_one(self.inputs_shape[1])
                self.bias_param['axis'] = compute_scale_axis(self.inputs_shape[0], self.inputs_shape[1])

                # Pre-Layer: Reshape
                self.pre_input = self.inputs[1]
                self.inputs[1] = 'reshape' + str(self.index)
                self.reshape_param = dict(shape=dict(dim=self.inputs_shape[1]))

                self.attrs = self.bias_param
        else:
            # Scale Layer
            self.layer_type = 'Scale'

            if self.inputs_buf[0] is not None:
                bias_index = 0
                input_index = 1
            else:
                bias_index = 1
                input_index = 0

            # Weight
            self.weight = np.ones(self.inputs_shape[bias_index], dtype=float, order='C')

            # Bias
            self.bias = self.inputs_buf[bias_index]

            # Attributes
            self.scale_param = dict()
            self.scale_param['bias_term'] = True

            # Axis
            if self.bias.shape != () and self.bias.shape != []:
                self.scale_param['axis'] = self.inputs_shape[input_index].index(self.bias.shape[0])
                self.scale_param['num_axes'] = len(self.bias.shape)
            else:
                self.scale_param['num_axes'] = 0

            self.attrs = self.scale_param

        self.setParsed()


    def convert(self):
        pre_layer = None
        if self.type == 'Eltwise':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)
        elif self.type == 'Scale':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)
        elif self.type == 'Bias':
            pre_layer = caffe_layer('Reshape', 'Reshape'+str(self.index), [self.pre_input], [None], ['reshape'+str(self.index)], reshape_param=self.reshape_param)
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, bias_param=self.bias_param)

        self.setConverted()

        return [layer] if pre_layer is None else [pre_layer, layer]

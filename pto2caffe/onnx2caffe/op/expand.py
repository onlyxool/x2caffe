import numpy as np

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Expand(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Expand')
        self.setInited()


    def parse(self):
        super().__parse__()

        input_shape = self.inputs_shape[0]
        output_shape = self.inputs_buf[1].tolist() if self.inputs_buf[1] is not None else self.outputs_shape[0]

        axes = np.nonzero(np.array(input_shape) - np.array(output_shape))[0].tolist()

        if self.inputs_buf[0] is not None:
            self.saveConstant(self.outputs[0], self.inputs_buf[0] * np.ones(output_shape))
        elif input_shape == output_shape:
            self.byPassOperator()
        elif len(axes) == 1:
            self.type = 'Tile'

            self.tile_param = dict()
            self.tile_param['axis'] = axes[0]
            self.tile_param['tiles'] = int((np.array(output_shape)/np.array(input_shape))[axes[0]])

            self.attrs = self.tile_param

            self.setParsed()
        elif len(axes) == 2:
            self.type = 'Tile+Tile'

            self.tile_param0 = dict()
            self.tile_param0['axis'] = axes[0]
            self.tile_param0['tiles'] = int((np.array(output_shape)/np.array(input_shape))[axes[0]])

            self.tile_param1 = dict()
            self.tile_param1['axis'] = axes[1]
            self.tile_param1['tiles'] = int((np.array(output_shape)/np.array(input_shape))[axes[1]])

            self.attrs = self.tile_param0

            self.setParsed()
        elif len(axes) == 3:
            self.type = 'Tile+Tile+Tile'

            self.tile_param0 = dict()
            self.tile_param0['axis'] = axes[0]
            self.tile_param0['tiles'] = int((np.array(output_shape)/np.array(input_shape))[axes[0]])

            self.tile_param1 = dict()
            self.tile_param1['axis'] = axes[1]
            self.tile_param1['tiles'] = int((np.array(output_shape)/np.array(input_shape))[axes[1]])

            self.tile_param2 = dict()
            self.tile_param2['axis'] = axes[2]
            self.tile_param2['tiles'] = int((np.array(output_shape)/np.array(input_shape))[axes[2]])

            self.attrs = self.tile_param0

            self.setParsed()
        else:
            raise NotImplementedError


    def convert(self):
        layers = list()
        if self.type == 'Tile':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, tile_param=self.tile_param))
        elif self.type == 'Tile+Tile':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], self.inputs, self.inputs_buf, self.interblob, tile_param=self.tile_param0))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], self.interblob, self.inputs_buf, self.outputs, tile_param=self.tile_param1))
        elif self.type == 'Tile+Tile+Tile':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], self.inputs, self.inputs_buf, [self.interblob[0]], tile_param=self.tile_param0))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], [self.interblob[0]], self.inputs_buf, [self.interblob[1]], tile_param=self.tile_param1))
            layers.append(caffe_layer(self.layer_type[2], self.name[2], [self.interblob[1]], self.inputs_buf, self.outputs, tile_param=self.tile_param2))

        self.setConverted()

        return layers

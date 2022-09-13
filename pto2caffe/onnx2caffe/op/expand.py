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

        if input_shape == output_shape:
            self.byPassOperator()
        elif len(axes) == 1:
            self.layer_type = 'Tile'

            self.tile_param = dict()
            self.tile_param['axis'] = axes[0]
            self.tile_param['tiles'] = int((np.array(output_shape)/np.array(input_shape))[axes[0]])

            self.attrs = self.tile_param

            self.setParsed()
        elif len(axes) == 2:
            self.layer_type = 'Tile+Tile'
            self.inter_blob = 'tile_tile_split' + str(self.index)

            self.tile_param0 = dict()
            self.tile_param0['axis'] = axes[0]
            self.tile_param0['tiles'] = int((np.array(output_shape)/np.array(input_shape))[axes[0]])

            self.tile_param1 = dict()
            self.tile_param1['axis'] = axes[1]
            self.tile_param1['tiles'] = int((np.array(output_shape)/np.array(input_shape))[axes[1]])

            self.attrs = self.tile_param0

            self.setParsed()
        else:
            print(self.attrs, axes, self.inputs_shape, self.outputs_shape)
            raise NotImplementedError


    def convert(self):
        layers = list()
        if self.type == 'Tile':
            layers.append(caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, tile_param=self.tile_param))
        elif self.type == 'Tile+Tile':
            layers.append(caffe_layer('Tile', self.name+'_0', self.inputs, self.inputs_buf, [self.inter_blob], tile_param=self.tile_param0))
            layers.append(caffe_layer('Tile', self.name+'_1', [self.inter_blob], self.inputs_buf, self.outputs, tile_param=self.tile_param1))

        self.setConverted()

        return layers

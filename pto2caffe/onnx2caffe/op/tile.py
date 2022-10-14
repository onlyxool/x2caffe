import numpy as np

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Tile(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Tile')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            self.saveConstant(self.outputs[0], np.tile(self.inputs_buf[0], self.inputs_buf[1]))
        elif self.inputs_shape[0] == self.outputs_shape[0]:
            self.byPassOperator()
        else:
            if self.inputs_buf[1] is not None:
                axes = np.nonzero(self.inputs_buf[1] - np.ones(self.inputs_buf[1].shape))[0].tolist()
            elif self.inputs_shape[0] is not None and self.outputs_shape[0] is not None:
                axes = np.nonzero(np.array(self.outputs_shape[0]) - np.array(self.inputs_shape[0]))[0].tolist()
            else:
                raise NotImplementedError

            if len(axes) == 1:
                self.type = 'Tile'
                self.tile_param = dict()
                self.tile_param['axis'] = axes[0]
                self.tile_param['tiles'] = int(self.inputs_buf[1][axes[0]]) if self.inputs_buf[1] is not None else int((np.array(self.outputs_shape[0]) / np.array(self.inputs_shape[0]))[axes[0]])

                self.attrs = self.tile_param

                self.setParsed()
            else:
                raise NotImplementedError



    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, tile_param=self.tile_param)

        self.setConverted()

        return [layer]

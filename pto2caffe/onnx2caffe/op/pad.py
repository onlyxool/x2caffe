import sys
import numpy as np

from onnx2caffe.op.operator import Operator


class Pad(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.pad = dict()
        self.setInited()


    def parse(self):
        self.layer_type = 'Pad'
        super().__parse__()

        if self.attrs.get('value', 0.0) != 0.0 or self.attrs.get('mode', b'constant').decode() != 'constant':
            errorMsg = 'Caffe support constant Pad mode only.'
            print('Warning:', errorMsg)
#            sys.exit(errorMsg)

        if self.model.opset[0] >= 11:
            pad = self.inputs_buf[1].reshape(-1, len(self.inputs_shape[0]))
        else:
            pad = np.array(self.attrs['pads']).reshape(-1, len(self.inputs_shape[0]))

        if len(self.inputs_shape[0]) == 4:
            self.pad['left']    = pad[0][3]
            self.pad['right']   = pad[1][3]
            self.pad['top']     = pad[0][2]
            self.pad['bottom']  = pad[1][2]
        else:
            errorMsg = 'Input tensor has' + len(self.inputs_shape[0]) + 'dimentions'
            sys.exit(errorMsg)

        self.isLegacy = True

    def convert(self):
        pass

import logging
import numpy as np

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')

class Pad(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.pad = dict()
        self.setInited()


    @property
    def type(self):
        return 'Pad'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Option
        self.parseAttributes()
        if self.attrs.get('value', 0.0) != 0.0 or self.attrs['mode'].decode() != 'constant':
            print('Warning: Caffe support constant Pad mode only.')
#            raise NotImplementedError('Caffe support constant Pad mode only.')

        if self.model.opset[0] >= 11:
            pad = self.inputs_buf[1].reshape(-1, len(self.inputs_shape[0]))
        else:
            pad = np.array(self.attrs['pads']).reshape(-1, len(self.inputs_shape[0]))

        if len(self.inputs_shape[0]) == 4:
            self.pad['left'] = pad[0][3]
            self.pad['right'] = pad[1][3]
            self.pad['top'] = pad[0][2]
            self.pad['bottom'] = pad[1][2]
        else:
            errorMsg = 'Input tensor has' + len(self.inputs_shape[0]) + 'dimentions'
            raise NotImplementedError(errorMsg)

        self.isLegacy = True

    def convert(self):
        pass

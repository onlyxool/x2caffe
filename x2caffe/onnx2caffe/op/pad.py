import numpy as np

from onnx2caffe.op.operator import Operator


class Pad(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Pad')
        self.setInited()


    def parse(self):
        self.type = 'Pad'
        super().__parse__()

        if self.attrs.get('value', 0.0) != 0.0 or self.attrs.get('mode', b'constant').decode() != 'constant':
            print('Warning:', 'Caffe support constant Pad mode only.')

        if self.model.opset[0] >= 11:
            pad = self.inputs_buf[1].reshape(-1, len(self.inputs_shape[0]))
        else:
            pad = np.array(self.attrs['pads']).reshape(-1, len(self.inputs_shape[0]))

        self.byPassOperator()

        if np.count_nonzero(pad) > 0:
            pad_dict = dict()
            if len(self.inputs_shape[0]) == 4:
                pad_dict['left']    = pad[0][3]
                pad_dict['right']   = pad[1][3]
                pad_dict['top']     = pad[0][2]
                pad_dict['bottom']  = pad[1][2]
                self.model.pad[self.outputs[0]] = pad_dict
            else:
                self.unSupported('Input tensor has' + str(len(self.inputs_shape[0])) + 'dimentions')


    def convert(self):
        pass

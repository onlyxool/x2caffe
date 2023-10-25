import numpy as np

from torch2caffe.op.operator import Operator


class ConstantPad(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'constant_pad_nd')
        self.setInited()


    def parse(self):
        self.type = 'ConstantPad'
        super().__parse__()

        pad = self.inputs_buf[1]
        value = self.inputs_buf[2]

        if value != 0:
            print('Warning:', 'Caffe support constant Pad mode only.')

        self.byPassOperator()

        if np.count_nonzero(pad) > 0:
            if len(pad) < 4:
                pad = [0] * (4 - len(pad)) + pad
            elif len(pad) > 4:
                self.unSupported('Input tensor has' + str(len(self.inputs_shape[0])) + 'dimentions')

            pad_dict = dict() # NCHW
            pad_dict['left']    = pad[0]
            pad_dict['right']   = pad[1]
            pad_dict['top']     = pad[2]
            pad_dict['bottom']  = pad[3]
            self.model.pad[self.outputs[0]] = pad_dict


    def convert(self):
        pass

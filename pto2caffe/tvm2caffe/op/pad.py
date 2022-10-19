import numpy as np

from tvm2caffe.op.operator import Operator


class Pad(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'nn.pad')
        self.setInited()


    def parse(self):
        self.type = 'Pad'
        super().__parse__()

        if self.attrs.get('pad_mode', 'constant') != 'constant':
            print('Warning: Caffe support constant Pad mode only.')

        pad_value = self.inputs_buf[1]
        if int(pad_value) != 0:
            print('Warning: Caffe can\'t support Pad_value = %d.'%pad_value)

        self.byPassOperator()

        if np.count_nonzero(np.array(self.attrs['pad_width'])) > 0:
            pad_dict = dict()
            if len(self.attrs['pad_width']) == 4:
                pad_dict['left']    = self.attrs['pad_width'][self.ndim('W')][0]
                pad_dict['right']   = self.attrs['pad_width'][self.ndim('W')][1]
                pad_dict['top']     = self.attrs['pad_width'][self.ndim('H')][0]
                pad_dict['bottom']  = self.attrs['pad_width'][self.ndim('H')][1]
                self.model.pad[self.outputs[0]] = pad_dict
            else:
                self.unSupported('Pad has' + str(len(self.attrs['pad_width'])) + 'dimentions')


    def convert(self):
        pass

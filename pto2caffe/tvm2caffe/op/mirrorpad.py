import numpy as np

from tvm2caffe.op.operator import Operator


class MirrorPad(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'nn.mirror_pad')
        self.setInited()


    def parse(self):
        self.type = 'Pad'
        super().__parse__()

        print('Warning: Caffe support constant Pad mode only.')

        self.byPassOperator()

        if np.count_nonzero(np.array(self.attrs['pad_width'])) > 0:
            pad_dict = dict()
            if len(self.attrs['pad_width']) == 4:
                # t, l, b, r
                pad = [self.attrs['pad_width'][self.ndim('H')][0], self.attrs['pad_width'][self.ndim('W')][0], self.attrs['pad_width'][self.ndim('H')][1], self.attrs['pad_width'][self.ndim('W')][1]]
                self.model.pad[self.outputs[0]] = pad
            else:
                self.unSupported('Pad has' + str(len(self.attrs['pad_width'])) + 'dimentions')


    def convert(self):
        pass

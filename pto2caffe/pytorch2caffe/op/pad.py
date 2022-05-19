from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator


class Pad(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        assert(self.operator_code == 'F.pad')
        self.setInited()


    def parse(self):
        self.layer_type = 'Pad'
        super().__parse__()

        if float(self.attrs.get('value', 0.0)) != 0.0 or self.attrs['mode'] != 'constant':
            errorMsg = 'Caffe support constant Pad mode only.'
            print('Warning:', errorMsg)
            #sys.exit(errorMsg)

        # Attributes
        self.pad = dict()
        self.pad['left']    = self.attrs['pad'][0]
        self.pad['right']   = self.attrs['pad'][1]
        self.pad['top']     = self.attrs['pad'][2]
        self.pad['bottom']  = self.attrs['pad'][3]

        self.attrs.update(self.pad)

        self.isLegacy = True


    def convert(self):
        pass

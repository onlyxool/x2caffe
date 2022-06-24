from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator


class Mean(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        assert(self.operator_code == 'torch.mean')
        self.setInited()


    def parse(self):
        self.layer_type = 'Pooling'
        super().__parse__()

        #self.attrs['keepdim']
        if self.attrs['dim'] != [2,3]:
            errorMsg = 'ReduceMean\'s axis: ' + axis + ' Not support'
            sys.exit(errorMsg)

        # Attributes
        self.pooling_param = dict()
        self.pooling_param['pool'] = 1
        self.pooling_param['kernel_h'] = self.inputs_shape[0][2]
        self.pooling_param['kernel_w'] = self.inputs_shape[0][3]
        self.pooling_param['stride'] = 1
        self.pooling_param['ceil_mode'] = False

        self.attrs = self.pooling_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param)

        self.setConverted()

        return [layer]

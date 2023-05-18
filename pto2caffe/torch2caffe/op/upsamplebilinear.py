from torch.nn.functional import interpolate

from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class UpsampleBilinear(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'upsample_bilinear2d')
        self.setInited()


    def parse(self):
        self.type = 'Interp'
        super().__parse__()

        self.interp_param = dict()
        self.interp_param['align_corners'] = self.inputs_buf[2]
        self.interp_param['height'] = self.inputs_buf[1][0]
        self.interp_param['width'] = self.inputs_buf[1][1]
        self.attrs = self.interp_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs[:1], self.inputs_buf, self.outputs, interp_param=self.interp_param)

        self.setConverted()

        return [layer]


    def forward(self):
        return interpolate(self.model.variable[self.inputs[0]], size=self.inputs_buf[1],
                scale_factor=None, mode='bilinear', align_corners=self.inputs_buf[2], recompute_scale_factor=None)

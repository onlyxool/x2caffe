import tflite

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator


class ResizeBilinear(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)

        assert(self.operator == 'RESIZE_BILINEAR')
        assert(self.op.InputsLength() == 2), "TFLite has only two inputs"
        assert(self.op.OutputsLength() == 1)

        self.setInited()


    def parse(self):
        self.parseInputOutput()

        # Attributes
        self.layer_type = 'Interp'
        op_opt = self.op.BuiltinOptions()
        opt = tflite.ResizeBilinearOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        self.interp_param = dict()
        self.interp_param['align_corners'] = opt.AlignCorners()
        self.interp_param['height'] = self.inputs_buf[1][0]
        self.interp_param['width'] = self.inputs_buf[1][1]

        self.attrs = self.interp_param
        #opt.HalfPixelCenters()

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, interp_param=self.interp_param)

        self.setConverted()

        return [layer]

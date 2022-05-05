import logging
from base import Base
from util import shape_map_nhwc2nchw

logger = logging.getLogger('TensorFlow2caffe')


class Operator(Base):

    def __init__(self, model, tf_op, op_type, index):
        super().__init__(model, model.graph, index)
        self.op = tf_op
        self.op_code = op_type
        self.inputs = []
        self.inputs_shape = []
        self.inputs_buf = []
        self.outputs = []
        self.outputs_shape = []
        self.pre = []  # ops that before this op which to enable TensorFlow op
        self.post = []  # ops that after this op which to enable TensorFlow op
        self.attrs = dict()


    @property
    def type(self):
        raise NotImplementedError("Method Operator.type() must be overrided!")


    @property
    def name(self):
        return self.type + str(self.index)


    @property
    def shorty(self):
        return '[%s](%s)' % (self.name, self.type)


    def str(self):
        return '[' + self.name + '] (' + self.type + ')'


    def __str__(self):
        inames = str([t for t in self.inputs])
        onames = str([t for t in self.outputs])
        return '%s attr%s: %s -> %s' % (self.shorty, self.attrs, inames, onames)


    def ndim(self, dim):
        return self.layout.find(dim)


    def parseInput(self):
        for i in range(len(self.op.inputs)):
            self.inputs.append(self.model.indentity.get(self.op.inputs[i].name, self.op.inputs[i].name))
            self.inputs_shape.append(shape_map_nhwc2nchw(self.op.inputs[i].shape))
            self.inputs_buf.append(self.model.constant.get(self.inputs[i], None))


    def parseOutput(self):
        for i in range(len(self.op.outputs)):
            self.outputs.append(self.op.outputs[i].name)
            self.outputs_shape.append(shape_map_nhwc2nchw(self.op.outputs[i].shape))


    def convertAttributeProto(self, attr, name):
        if attr[name].HasField('b'):
            return attr[name].b
        elif attr[name].HasField('f'):
            return attr[name].f
        elif attr[name].HasField('func'):
            return attr[name].func
        elif attr[name].HasField('i'):
            return attr[name].i
        elif attr[name].HasField('list'):
            return list(attr[name].list.i)
        elif attr[name].HasField('s'):
            return attr[name].s.decode('utf-8')
        elif attr[name].HasField('shape'):
            return attr[name].shape
        elif attr[name].HasField('tensor'):
            return attr[name].tensor
        elif attr[name].HasField('type'):
            return attr[name].type
        else:
            raise ValueError("Unsupported TensorFlow attribute: {}".format(attr))


    def parseAttributes(self):
        for op_attr in self.op.op_def.attr:
            self.attrs[op_attr.name] = self.convertAttributeProto(self.op.node_def.attr, op_attr.name)
        self.layout = self.attrs.get('data_format', 'NHWC')

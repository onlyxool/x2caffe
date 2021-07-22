import numpy as np
from base import Base
from onnx import numpy_helper
#from util import *

class Operator(Base):
    def __init__(self, model, node, index):
        super().__init__(model, model.graph, index)
        self.node = node
        self.op_code = node.op_type
        self.name = self.type + str(index)
        self.index = index
        self.inputs = []
        self.inputs_shape = []
        self.inputs_buf = []
        self.outputs = []
        self.outputs_shape = []
        self.pre = []  # ops that before this op which to enable TFLite op
        self.post = []  # ops that after this op which to enable TFLite op
        self.attrs = dict()


    @property
    def type(self):
        raise NotImplementedError("Method Operator.type() must be overrided!")


    def propagatableTensors(self):
        """Get all layout propagable tensors of this operator.

        When we propagate layouts across the graph:
        1. Some operators may stop the propagation
            a) An operator assumes layouts of its tensors, `Conv` for example.
               Such operator needs to define the layouts of its tensors explicitly.
            b) An operator breaks layout semantic, `Reshape` for example.
               Tensors connected to this operator should be propagated.
               And the operator may need special handling regarding layout.
        2. Others may not - propagatable:
            a) An operator that is transparent to layout, such as Add.
               Just propagate the layouts.
            b) Layout can propagate across tensors of an operator, but the operator
               itself has attribution that is sensitive to layout.
               Operator needs special handling after propagation.
        This is defined per operator.

        To handle this, we firstly propagate layouts of tensors across the graph,
        and then update attributes of operators accordingly.
        """
        raise NotImplementedError("Method %s.propagatableTensors() must be overrided!" % self.type)


    def transform(self):
        """Transform the operator attributions w.r.t. propagated layouts.

        The attributions could be a tensor that describing layout related things.
        Operators that defined as 1.a, 1.b and 2.b in `layoutPropagatable()`
        are such cases. But not all of them need special treatment.
        For example, `Conv` doesn't need additional processing after propagation.

        This must be called after the layouts have been propagated across graph.
        """
        raise NotImplementedError("Method %s.transform() must be overrided!" % self.type)


    def convertAttributeProto(self, attr):
        if attr.HasField('f'):
            return attr.f
        elif attr.HasField('i'):
            return attr.i
        elif attr.HasField('s'):
            return attr.s
        elif attr.HasField('t'):
            return numpy_helper.to_array(attr.t)
        elif len(attr.floats):
            return list(attr.floats)
        elif len(attr.ints):
            return list(attr.ints)
        elif len(attr.strings):
            return list(attr.strings)
        else:
            raise ValueError("Unsupported ONNX attribute: {}".format(attr))



    def parseInput(self):
        for input in self.node.input:
            self.inputs.append(input)
            self.inputs_buf.append(self.model.input_tensor.get(input, None))
            if input in self.model.shape:
                self.inputs_shape.append(self.model.shape[input])
            elif input in self.model.input_tensor:
                self.inputs_shape.append(self.model.input_tensor[input].shape)
            else:
                self.inputs_shape.append(None)


    def parseOutput(self):
        for output in self.node.output:
            self.outputs.append(output)
            self.outputs_shape.append(self.model.shape[output])


    def parseAttributes(self):
        for attr in self.node.attribute:
            self.attrs[attr.name] = self.convertAttributeProto(attr)


    @property
    def type(self):
        raise NotImplementedError("Method Operator.type() must be overrided!")

    @property
    def shorty(self):
        return '[%s](%s)' % (self.name, self.type)

    def str(self):
        return '[' + self.name + '] (' + self.type + ')'

    def __str__(self):
        inames = str([t for t in self.inputs])
        onames = str([t for t in self.outputs])
        return '%s attr%s: %s -> %s' % (self.shorty, self.attrs, inames, onames)


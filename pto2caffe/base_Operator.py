import numpy as np

from base import Base
from util import isShapeCompatible

class BaseOperator(Base):

    def __init__(self, model, graph, index):
        super().__init__(model, graph, index)
        self.layout = model.layout

        self.operator_code = None
        self.type = None
        self.weight = None
        self.bias = None
        self.attrs = dict()
        self.layers = list()


    @property
    def layer_type(self):
        if self.type is not None and self.type.find('+') >= 0:
            return self.type.split('+')
        elif self.type is not None:
            return self.type
        else:
            return self.operator_code


    @property
    def name(self):
        if self.type is not None and self.type.find('+') >= 0:
            return [layer_type + str(self.index) + '_' + str(index) for index, layer_type in enumerate(self.type.split('+'))]
        elif self.layer_type is not None:
            return self.layer_type + str(self.index)


    @property
    def shorty(self):
        return '[%s](%s)' % (str(self.name), str(self.type))


    def str(self):
        return '[' + str(self.name) + ']  (' + str(self.type) + ')'


    @property
    def attrs2str(self):
        attrstr = ''
        for key, value in self.attrs.items():
            attrstr = attrstr + '    ' + str(key) + ': ' + str(value) + '\n'
        return attrstr


    def __str__(self):
        inames = str([t for t in self.inputs])
        onames = str([t for t in self.outputs])
        ishape = str(self.inputs_shape)
        oshape = str(self.outputs_shape)
        inbufs = str([None if t is None else 'np.array' for t in self.inputs_buf])
        return '\n%s\n%s    %s -> %s\n    %s -> %s\n    %s\n' % (self.shorty, self.attrs2str, inames, onames, ishape, oshape, inbufs)


    def ndim(self, dim):
        return self.layout.index(dim)


    def checkShapeCompatible(self):
        if not isShapeCompatible(self.inputs_shape[0], self.inputs_shape[1]) and self.inputs_buf[1] is None:
            self.inputs_shape[1] = list(np.squeeze(np.random.random(self.inputs_shape[1])).shape)
            if not isShapeCompatible(self.inputs_shape[0], self.inputs_shape[1]):
                return False
            return 'Squeeze'
        elif not isShapeCompatible(self.inputs_shape[0], self.inputs_shape[1]):
            self.inputs_buf[1] = np.squeeze(self.inputs_buf[1])
            self.inputs_shape[1] = list(self.inputs_buf[1].shape)
            if not isShapeCompatible(self.inputs_shape[0], self.inputs_shape[1]):
                return False

        return True


    @property
    def interblob(self):
        if self.type is not None and self.type.find('+') >= 0:
            return ['intermediate_' + str(self.index) + '_' + str(index) for index in range(self.type.count('+'))]
        elif self.type is not None:
            raise ValueError


    def inputBuf_byName(self, name):
        for index, input in enumerate(self.inputs):
            if input.find(name) > 0 or input.find(name.lower()) > 0 or input.find(name.upper()) > 0:
                return self.inputs_buf[index]


    def input_byName(self, name):
        for index, input in enumerate(self.inputs):
            if input.find(name) > 0 or input.find(name.lower()) > 0 or input.find(name.upper()) > 0:
                return input


    def saveConstant(self, name, constant):
        self.type = 'Constant'
        self.model.constant[name] = constant


    def byPassOperator(self):
        raise NotImplementedError("method byPassOperator() should be overrided!")


    def unSupported(self, errorMsg=None):
        raise NotImplementedError("method unSupported() should be overrided!")


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

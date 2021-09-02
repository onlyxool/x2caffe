import tflite
import logging
import numpy as np
from base import Base
from util import *

logger = logging.getLogger('tflite2caffe')

class Operator(Base):

    def __init__(self, model, tf_op:tflite.Operator, tf_op_code, index):
        super().__init__(model, model.graph, index)
        self.op = tf_op
        self.op_code = tf_op_code
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


    def getBuffer(self, tensor_id):
        type_id = self.graph.Tensors(tensor_id).Type()
        tensor_type = ['float32', 'float16', 'int32', 'uint8', 'int64', 'string', 'bool', 'int16', 'COMPLEX64', 'int8', 'float64', 'COMPLEX128']
        if (tensor_type[type_id] not in ['int32', 'float32', 'uint8', 'int8', 'float16']):
            logger.warning("Data type {} not supported/tested yet, "
                       "the generated model may contain error".format(tensor_type[type_id]))

        assert(tensor_id < self.graph.TensorsLength())

        tensor = self.graph.Tensors(tensor_id)
        tensor_buf = tensor.Buffer()
        shape = tensor.ShapeAsNumpy()
        assert(tensor_buf < self.model.model.BuffersLength())

        raw = self.model.model.Buffers(tensor_buf).DataAsNumpy()
        if isinstance(raw, int) and raw == 0:
            return None

        data = np.frombuffer(raw, dtype=tensor_type[type_id])

        if isinstance(shape, np.ndarray) and len(shape) > 0:
            data = data.reshape(shape)

        return data.copy()


    def parseInput(self):
        for i in range(self.op.InputsLength()):
            if self.op.Inputs(i) >= 0:
                self.inputs.append(self.op.Inputs(i))
                self.inputs_shape.append(shape_map_nhwc2nchw(self.graph.Tensors(self.inputs[i]).ShapeAsNumpy()))
                buf = self.getBuffer(self.op.Inputs(i))
                self.inputs_buf.append(buf)
            else:
                self.inputs_buf.append(None)


    def parseOutput(self):
        for i in range(self.op.OutputsLength()):
            if self.op.Outputs(i) >= 0:
                self.outputs.append(self.op.Outputs(i))
                self.outputs_shape.append(shape_map_nhwc2nchw(self.graph.Tensors(self.outputs[0]).ShapeAsNumpy()))


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

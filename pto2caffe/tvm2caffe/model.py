import re
import sys
import tvm
import logging
from base_Model import BaseModel

from tvm2caffe.op.abs import Abs
from tvm2caffe.op.add import Add
from tvm2caffe.op.exp import Exp
from tvm2caffe.op.log import Log
from tvm2caffe.op.max import Max
from tvm2caffe.op.pad import Pad
from tvm2caffe.op.sum import Sum
from tvm2caffe.op.bias import Bias
from tvm2caffe.op.clip import Clip
from tvm2caffe.op.copy import Copy
from tvm2caffe.op.relu import ReLU
from tvm2caffe.op.sqrt import Sqrt
from tvm2caffe.op.tanh import Tanh
from tvm2caffe.op.dense import Dense
from tvm2caffe.op.power import Power
from tvm2caffe.op.prelu import PReLU
from tvm2caffe.op.split import Split
from tvm2caffe.op.argmax import Argmax
from tvm2caffe.op.concat import Concat
from tvm2caffe.op.divide import Divide
from tvm2caffe.op.resize import Resize
from tvm2caffe.op.maximum import Maximum
from tvm2caffe.op.minimum import Minimum
from tvm2caffe.op.pooling import Pooling
from tvm2caffe.op.reshape import Reshape
from tvm2caffe.op.sigmoid import Sigmoid
from tvm2caffe.op.softmax import Softmax
from tvm2caffe.op.multiply import Multiply
from tvm2caffe.op.negative import Negative
from tvm2caffe.op.subtract import Subtract
from tvm2caffe.op.upsample import Upsample
from tvm2caffe.op.transpose import Permute
from tvm2caffe.op.batchnorm import BatchNorm
from tvm2caffe.op.mirrorpad import MirrorPad
from tvm2caffe.op.reducemean import ReduceMean
from tvm2caffe.op.convolution import Convolution
from tvm2caffe.op.spacetodepth import SpaceToDepth
from tvm2caffe.op.stridedslice import StridedSlice
from tvm2caffe.op.convtranspose import ConvTranspose
from tvm2caffe.op.bypassoperator import ByPassOperator

from util import shape_map_nhwc2nchw
from caffe_transform import make_caffe_input_layer
from tvm2caffe.relay import preprocess, get_relay_type, get_tensor_shape, remove_numTypeExt


OpMap = {
    'abs': Abs,
    'add': Add,
    'exp': Exp,
    'log': Log,
    'max': Max,
    'sum': Sum,
    'clip': Clip,
    'copy': Copy,
    'sqrt': Sqrt,
    'tanh': Tanh,
    'nn.pad': Pad,
    'power': Power,
    'split': Split,
    'nn.relu': ReLU,
    'argmax': Argmax,
    'divide': Divide,
    'nn.dense': Dense,
    'nn.prelu': PReLU,
    'mean': ReduceMean,
    'maximum': Maximum,
    'minimum': Minimum,
    'reshape': Reshape,
    'squeeze': Reshape,
    'sigmoid': Sigmoid,
    'nn.bias_add': Bias,
    'multiply': Multiply,
    'negative': Negative,
    'subtract': Subtract,
    'transpose': Permute,
    'nn.softmax': Softmax,
    'nn.leaky_relu': ReLU,
    'concatenate': Concat,
    'cast': ByPassOperator,
    'expand_dims': Reshape,
    'array': ByPassOperator,
    'nn.conv2d': Convolution,
    'nn.avg_pool2d': Pooling,
    'nn.max_pool2d': Pooling,
    'image.resize2d': Resize,
    'nn.upsampling': Upsample,
    'nn.batch_norm': BatchNorm,
    'nn.mirror_pad': MirrorPad,
    'strided_slice': StridedSlice,
    'nn.global_avg_pool2d': Pooling,
    'nn.space_to_depth': SpaceToDepth,
    'nn.conv2d_transpose': ConvTranspose,
}


logger = logging.getLogger('Tvm2Caffe')


class Model(BaseModel):

    def __init__(self, tvm_model, tvm_model_params, param):
        required_pass=['RemoveUnusedFunctions', 'ConvertLayout', 'FoldConstant', 'InferType', 'SimplifyInference', 'CombineParallelConv2D', 'FoldScaleAxis', 'ForwardFoldScaleAxis']
        disabled_pass=['FuseOps']
        with tvm.transform.PassContext(opt_level=0, required_pass=required_pass, disabled_pass=disabled_pass):
            model = tvm.relay.optimize(tvm_model,'llvm', params=tvm_model_params)

        required_pass_tir='RemoveNoOp'
        disabled_pass_tir=['UnrollLoop', 'VectorizeLoop', 'tir.SkipAssert', 'tir.ThreadSync', 'tir.Apply', 'tir.CoProcSync', 'tir.ConvertForLoopsToSerial']
        with tvm.transform.PassContext(opt_level=0, required_pass=required_pass_tir, disabled_pass=disabled_pass_tir, config={"relay.FuseOps.max_depth": 1}):
            lib = tvm.relay.build(model[0], target='llvm')

        model_txt = model[0].astext(show_meta_data=False)
        super().__init__(model[0], model_txt[model_txt.find('main'):model_txt.rfind('}')].strip(), param)


        self.device = tvm.device('llvm', 0)
#        self.runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](self.device))
        self.runtime = tvm.contrib.debugger.debug_executor.GraphModuleDebug(lib["debug_create"](lib.libmod_name, self.device), [self.device], lib.graph_json, dump_root=None)

        self.relays = list()
        self.debug_outputs = dict(zip(OpMap.keys(), [None]*len(OpMap.keys())))
        self.outputs_buf = dict(zip(OpMap.keys(), [None]*len(OpMap.keys())))

        self.setInited()


    def preprocess(self, relays):
        for index, relay in enumerate(relays[1:]):
            relay_type = get_relay_type(relay)
            output = str(index) if relay.strip().startswith(relay_type) else re.compile(r'%(.+?) ').findall(relay.split('= ')[0])[0]
            shape_str = get_tensor_shape(relay.split(') /*')[-1])
            shape_str = remove_numTypeExt(shape_str[0] if len(shape_str) > 0 else None)
            output_shape = eval('['+shape_str+']')# if len(shape_str) > 0 else None

            if relay_type == '':
                self.relays.append('ignore')
                inputs = re.compile(r'%(.+?),').findall(relay.split(' = ')[-1].split(') /*')[0]+',')
                inputs_shape = get_tensor_shape(relay.split(') /*')[-1])
                self.indentity[output] = inputs
                for index, input_name in enumerate(inputs):
                    self.tensor_shape[input_name] = list(eval(remove_numTypeExt(inputs_shape[index])))
            elif relay_type.startswith('ty=Tensor'):
                self.relays.append('ignore')
                input = re.compile(r'%(.+?)\.').findall(relay.split(' = ')[-1])[0]
                no = re.compile(r'\.(.+?) ').findall(relay.split(' = ')[-1])[0]
                if input in self.indentity.keys():
                    self.indentity[input].append(output)
                else:
                    self.indentity[input] = [output]
            else:
                self.relays.append(relay)

            self.tensor_shape[output] = output_shape


    def parse(self):
        logger.debug("Parsing the TVM Model...")

        # Model Params
#        for key in self.model_params.keys():
#            self.constant[key] = self.model_params[key].numpy()
#            self.tensor_shape[key] = list(self.model_params[key].numpy().shape)

        # Meta Data
        if self.model.astext(show_meta_data=True).find('metadata') >= 0:
            metadata = tvm.ir.load_json(self.model.astext(show_meta_data=True).split('[metadata]')[-1])
            for index, meta in enumerate(metadata['relay.Constant']):
                self.constant['[relay.Constant]['+str(index)+']'] = meta.data.numpy()
                self.tensor_shape['[relay.Constant]['+str(index)+']'] = list(meta.data.numpy().shape) if self.layout == 'NCHW' else shape_map_nhwc2nchw(list(meta.data.numpy().shape))

        relays = self.graph.split('\n')

        if self.param['log'] == 0:
            print(self.graph)

        self.preprocess(relays)
        print('Tvm Model Input size:')
        inputs_str = relays[0][relays[0].find('main')+4:relays[0].rfind('hash')].strip()
        for inputs in inputs_str.split('(%')[-1].split(', %'):
            self.inputs.append(inputs.split(' ')[0])
            self.inputs_shape.append(eval('['+get_tensor_shape(inputs.split(': ')[-1])[0]+']'))
            self.inputs_dtype.append(re.compile(r'\), (.+?)\]').findall(inputs.split(': ')[-1])[0])

        for index, inputs_name in enumerate(self.inputs):
            print(inputs_name, end=':')
            print(self.inputs_shape[index], self.inputs_dtype[index])
            self.tensor_shape[inputs_name] = self.inputs_shape[index]

        for index, relay in enumerate(self.relays):
            relay = preprocess(relay)
            relay_type = get_relay_type(relay)

            if relay == 'ignore':
                continue

            if relay_type not in OpMap: # Unsupport OP
                if self.param['log'] == 1:
                    print(relay_type, relay)
                self.unsupport.append(relay_type)
                continue

            op = OpMap[relay_type](self, relay, index)
            op.parse()

            if op.status.parsed:
                self.operators.append(op)

        for errorMsg in list(set(self.errorMsg)):
            print(errorMsg)

        if len(self.unsupport) > 0:
            errorMsg = 'Error: Operator ' + str(list(set(self.unsupport))) + ' does not Support.\n'
            sys.exit(errorMsg)

        self.setParsed()


    def convert(self):
        logger.debug("Converting the Model...")

        for index, input_name in enumerate(self.inputs):
            input_shape = shape_map_nhwc2nchw(self.inputs_shape[index]) if self.layout == 'NHWC' else self.inputs_shape[index]
            self.layers.append(make_caffe_input_layer(input_name, input_shape, index, self.param))

        for op in self.operators:
            self.layers.extend(op.convert())

        self.setConverted()


    def forward(self, output_name, inputs_tensor):
        if not output_name.isdigit():
            return None

        if self.param['log'] == 0:
            print(self.runtime.benchmark(self.device))

        keys = list()
        if not self.status.forwarded:
            for index, input_name in enumerate(self.inputs):
                self.runtime.set_input(input_name, inputs_tensor[index].transpose(0, 2, 3, 1) if self.layout == 'NHWC' and len(self.inputs_shape[index]) == 4 else inputs_tensor[index])

            self.runtime.run()

            for (key, value) in self.runtime.debug_datum.get_output_tensors().items():
                if not key.startswith('p'):
                    key_type = key.split('tvmgen_default_fused_')[-1].split('____topo')[0].replace('_', '.', 1)
                    for op_type in OpMap.keys():
                        if key_type.startswith(op_type):
                            if self.outputs_buf[op_type] is None:
                                self.outputs_buf[op_type] = list()
                            self.outputs_buf[op_type].append(value.numpy())
                    keys.append(key)

            self.setForwarded()

        for (key, value) in self.debug_outputs.items():
            if self.outputs_buf[key] is not None and self.debug_outputs[key] is not None and output_name in self.debug_outputs[key]:
                output = self.outputs_buf[key][self.debug_outputs[key].index(output_name)]
                break
#        output = self.runtime.get_output(0).numpy()


        if 'output' not in locals():
            return None
        else:
            return output.transpose(0, 3, 1, 2) if self.layout == 'NHWC' and len(output.shape) == 4 else output

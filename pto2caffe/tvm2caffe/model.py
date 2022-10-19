import re
import sys
import tvm
import logging
from base import Base

from tvm2caffe.relay import preprocess, get_relay_type, get_tensor_shape

#from tvm2caffe.op.operator import Operator

from tvm2caffe.op.add import Add
from tvm2caffe.op.pad import Pad
from tvm2caffe.op.bias import Bias
from tvm2caffe.op.clip import Clip
from tvm2caffe.op.relu import ReLU
from tvm2caffe.op.dense import Dense
from tvm2caffe.op.concat import Concat
from tvm2caffe.op.resize import Resize
from tvm2caffe.op.reshape import Reshape
from tvm2caffe.op.softmax import Softmax
from tvm2caffe.op.multiply import Multiply
from tvm2caffe.op.transpose import Permute
from tvm2caffe.op.batchnorm import BatchNorm
from tvm2caffe.op.reducemean import ReduceMean
from tvm2caffe.op.convolution import Convolution
from tvm2caffe.op.bypassoperator import ByPassOperator

from caffe_transform import save_caffe_model
from caffe_transform import make_caffe_input_layer
from util import shape_map_nhwc2nchw


OpMap = {
    'array': ByPassOperator,
    'bypass': ByPassOperator,
    'concatenate': Concat,
    'add': Add,
    'transpose': Permute,
    'nn.pad': Pad,
    'nn.relu': ReLU,
    'nn.conv2d': Convolution,
    'nn.bias_add': Bias,
    'nn.softmax': Softmax,
    'nn.leaky_relu': ReLU,
    'image.resize2d': Resize,
    'nn.batch_norm': BatchNorm,
    'mean': ReduceMean,
    'reshape': Reshape,
    'clip': Clip,
    'nn.dense': Dense,
    'multiply': Multiply,
}


logger = logging.getLogger('Tvm2Caffe')


class Model(Base):

    def __init__(self, model, model_params, param):
        model_txt = model.astext(show_meta_data=False)
        super().__init__(model, model_txt[model_txt.find('main'):model_txt.rfind('}')].strip())

        with tvm.transform.PassContext(opt_level=0):
            lib = tvm.relay.build(model, target='llvm', params=model_params)
        self.device = tvm.device('llvm', 0)
        self.module = tvm.contrib.graph_executor.GraphModule(lib["default"](self.device))
#        self.module = tvm.contrib.debugger.debug_executor.GraphModuleDebug(lib['debug_create']('default', self.device), [self.device], lib.graph_json, None)

        self.param = param
        self.model_params = model_params
        self.layout = param['layout']

        self.inputs = list()
        self.inputs_shape = list()
        self.inputs_dtype = list()
        self.inputs_maxval = list()
        self.inputs_minval = list()

        self.outputs = list()
        self.outputs_shape = list()
        self.outputs_dtype = list()
        self.outputs_maxval = list()
        self.outputs_minval = list()

        self.pad = dict()
        self.layers = list()
        self.constant = dict()
        self.errorMsg = list()
        self.indentity = dict()
        self.operators = list()
        self.unsupport = list()
        self.tensor_shape = dict()

        self.setInited()


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
                self.tensor_shape['[relay.Constant]['+str(index)+']'] = list(meta.data.numpy().shape)

        relays = self.graph.split('\n')

        print('Tvm Model Input size:')
        inputs_str = relays[0][relays[0].find('main')+4:relays[0].rfind('hash')].strip()
        for inputs in inputs_str.split('(%')[-1].split(', %'):
            self.inputs.append(inputs.split(' ')[0])
            self.inputs_shape.append(eval('['+get_tensor_shape(inputs.split(': ')[-1])[0]+']'))
            self.inputs_dtype.append(re.compile(r'\), (.+?)\]').findall(inputs.split(': ')[-1])[0])

        for index, inputs_name in enumerate(self.inputs):
            print(inputs_name, end='')
            print(':', self.inputs_shape[index], self.inputs_dtype[index])
            self.tensor_shape[inputs_name] = self.inputs_shape[index]

        for index, relay in enumerate(relays[1:]):
            relay = preprocess(relay)
            relay_type = get_relay_type(relay)
            if relay_type not in OpMap: # Unsupport OP
                print(relay)
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


    def save(self, caffe_model_path):
        return save_caffe_model(caffe_model_path, self.layers)


    def forward(self, output_name, inputs_tensor):
        if self.param['log'] == 0:
            print(self.module.benchmark(self.device))

        for index, input_name in enumerate(self.inputs):
            self.module.set_input(input_name, inputs_tensor[index].transpose(0, 2, 3, 1) if self.layout == 'NHWC' and len(self.inputs_shape[index]) == 4 else inputs_tensor[index])

        self.module.run()

        output = self.module.get_output(0).numpy()
        return output.transpose(0, 3, 1, 2) if self.layout == 'NHWC' and len(output.shape) == 4 else self.module.get_output(0).numpy()



#        print(self.module.get_num_outputs(), 'get_num_outputs')
#        print(self.module.get_num_inputs(), 'get_num_inputs')
#        print(self.module.get_input_index('p35'), 'fuck')
#        print(self.module.get_input_info(), 'get_input_info')

#print(self.module.get_input(0), 'get_input0')
#print(dir(self.module), type(self.module))
#print(self.module.debug_get_output(520, tvm.nd.empty([1, 19, 128, 128])))
        #print(self.module.load_params())
        #print(self.module.share_params())


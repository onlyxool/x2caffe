import numpy as np

import caffe_path
import caffe
from caffe.proto import caffe_pb2


def param_name_dict():
    """Find out the correspondence between layer names and parameter names."""

    layer = caffe_pb2.LayerParameter()
    # get all parameter names (typically underscore case) and corresponding
    # type names (typically camel case), which contain the layer names
    # (note that not all parameters correspond to layers, but we'll ignore that)
    param_names = [f.name for f in layer.DESCRIPTOR.fields if f.name.endswith('_param')]
    param_type_names = [type(getattr(layer, s)).__name__ for s in param_names]

    # strip the final '_param' or 'Parameter'
    param_names = [s[:-len('_param')] for s in param_names]
    param_type_names = [s[:-len('Parameter')] for s in param_type_names]

    return dict(zip(param_type_names, param_names))


def assign_proto(proto, name, val):
    """Assign a Python object to a protobuf message, based on the Python
    type (in recursive fashion). Lists become repeated fields/messages, dicts
    become messages, and other types are assigned directly. For convenience,
    repeated fields whose values are not lists are converted to single-element
    lists; e.g., `my_repeated_int_field=3` is converted to
    `my_repeated_int_field=[3]`."""

    is_repeated_field = hasattr(getattr(proto, name), 'extend')
    if is_repeated_field and not isinstance(val, list):
        val = [val]

    if isinstance(val, list):
        if isinstance(val[0], dict):
            for item in val:
                proto_item = getattr(proto, name).add()
                for key, value in item.items():
                    assign_proto(proto_item, key, value)
        else:
            try:
                getattr(proto, name).extend(val)
            except:
                print('Value Error: Check Attribute data type, name:', name, ' value:', val)
                raise ValueError
    elif isinstance(val, dict):
        for key, value in val.items():
            assign_proto(getattr(proto, name), key, value)
    else:
        try:
            setattr(proto, name, val)
        except:
            print(proto, name, val)
            raise ValueError


class caffe_layer(object):
    def __init__(self, layer_type:str, layer_name:str, inputs:list, inputs_buf:list, outputs:list, weight=None, bias=None, **params):
        self.type = layer_type
        self.name = layer_name
        self.inputs = inputs
        self.inputs_buf = inputs_buf
        self.outputs = outputs
        self.params = params
        self.weight = weight
        self.bias = bias


    def _to_proto(self):
        layer = caffe_pb2.LayerParameter()
        layer.type = self.type
        layer.name = self.name

        # Bottom
        bottom_names = []
        for index, input_name in enumerate(self.inputs):
            if self.inputs_buf[index] is None:
                bottom_names.append(str(input_name))
        layer.bottom.extend(bottom_names)

        # Top
        top_names = []
        for top in self.outputs:
            top_names.append(str(top))
        layer.top.extend(top_names)

        # Param
        for key, value in self.params.items():
            if key.endswith('param'):
                assign_proto(layer, key, value)
            else:
                try:
                    assign_proto(getattr(layer, _param_names[self.type] + '_param'), key, value)
                except (AttributeError, KeyError):
                    assign_proto(layer, key, value)

        return layer


_param_names = param_name_dict()


def save_caffe_model(caffe_model_path, layers):
    proto = caffe_pb2.NetParameter()
    proto_layers = []
    for id, layer in enumerate(layers):
        proto_layers.append(layer._to_proto())
    proto.layer.extend(proto_layers)

    prototxt_save_path = caffe_model_path + '.prototxt'
    with open(prototxt_save_path, 'w') as f:
        print(proto, file=f)

    if caffe.__version__.find('gpu') >= 0:
        caffe.set_mode_gpu()
        caffe.set_device(0)
    else:
        caffe.set_mode_cpu()

    model = caffe.Net(prototxt_save_path, caffe.TEST)
    for id, layer in enumerate(layers):
        try:
            if layer.weight is not None:
                np.copyto(model.params[layer.name][0].data, layer.weight, casting='same_kind')
            if layer.bias is not None:
                np.copyto(model.params[layer.name][1].data, layer.bias, casting='same_kind')
            if layer.type == 'BatchNorm':
                np.copyto(model.params[layer.name][2].data, np.array([1.0]), casting='same_kind')
        except:
            raise Exception(layer.name)

    model_save_path = caffe_model_path + '.caffemodel'
    model.save(model_save_path)

    return model


dtype_map = {np.uint8: 0, np.int16: 1, np.float32: 2}

def make_caffe_input_layer(input, input_shape, index, param):
    layer_name = 'input' + str(index)

    image_data_param = dict()
    bin_data_param = dict()
    transform_param = dict()

    if param['source']:
        input_files = np.loadtxt(param['source'], dtype=str).tolist()
        if isinstance(input_files, list):
            ext = input_files[0].split('.')[-1].lower()
        elif isinstance(input_files, str):
            ext = input_files.split('.')[-1].lower()

        if ext in ['jpg', 'bmp', 'png', 'jpeg']:
            layer_type = 'ImageData'
            image_data_param['source'] = param['source']
            image_data_param['root_folder'] = param['root_folder'] + '/'
        elif ext == 'bin':
            layer_type = 'Input'
            bin_data_param['source'] = param['source']
            bin_data_param['root_folder'] = param['root_folder'] + '/'
            bin_data_param['data_format'] = dtype_map[param['dtype']]
            bin_data_param['shape'] = dict(dim=param['bin_shape'])
    else:
        ext = None
        layer_type = 'Input'

    if param['color_format'] == 'BGR':
        image_data_param['color_format'] = 0
    elif param['color_format'] == 'RGB':
        image_data_param['color_format'] = 1
    elif  param['color_format'] == 'GRAY':
        image_data_param['color_format'] = 2
        image_data_param['is_color'] = False

    if param['mean'] is not None:
        caffe_mean = np.array(param['scale']) * np.array(param['mean'])
        transform_param['mean_value'] = caffe_mean.tolist()

    if param['std'] is not None:
        caffe_scale = 1/(np.array(param['scale']) * np.array(param['std']))
        transform_param['scale'] = caffe_scale.tolist()

    if param.get('auto_crop', 0) == 1:
        if len(input_shape) == 4:
            param['crop_h'] = input_shape[2]
            param['crop_w'] = input_shape[3]

    if param['crop_h'] is not None:
        transform_param['crop_h'] = param['crop_h']
    if param['crop_w'] is not None:
        transform_param['crop_w'] = param['crop_w']

    if ext in ['jpg', 'bmp', 'png', 'jpeg']:
        return caffe_layer(layer_type, layer_name, [], [], [input], transform_param=transform_param, image_data_param=image_data_param, include=dict(phase=1))
    elif ext == 'bin':
        return caffe_layer(layer_type, layer_name, [], [], [input], transform_param=transform_param, \
                bin_data_param=bin_data_param, input_param=dict(shape=dict(dim=input_shape)), include=dict(phase=1))
    else:
        return caffe_layer(layer_type, layer_name, [], [], [input], transform_param=transform_param, \
                input_param=dict(shape=dict(dim=input_shape)), include=dict(phase=1))

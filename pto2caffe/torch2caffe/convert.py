import sys
import torch
from compare import compare
from torch2caffe.model import Model
from preprocess import gen_input_tensor


def convert(pytorch_file, caffe_model_path, param=None):
    try:
        torchscript = torch.jit.load(pytorch_file)
    except:
        sys.exit('Error: Model file is not Torchscript.\n')

    if hasattr(torchscript, 'training') and torchscript.training:
        torchscript.eval()

    if not isinstance(param['input_shape'], list):
        sys.exit('\nError: Dynamic Model input detected, Please Use -input_shape to overwrite input shape.\n')

    model = Model(torchscript, param)

    inputs_tensor = list()
    for index, input_name in enumerate(model.inputs):
        inputs_tensor.append(gen_input_tensor(param, model.inputs_shape[index], model.inputs_dtype[index], None))
        model.variable[input_name] = torch.Tensor(inputs_tensor[index])

    model.parse()
    model.convert()
    caffe_net = model.save(caffe_model_path)

    compare(model, caffe_net, inputs_tensor, param.get('compare', -1))

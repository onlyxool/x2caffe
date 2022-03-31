import sys
import torch
from compare import compare
from caffe_dump import dump_caffe_model
from pytorch2caffe.model import Model


def check_model_file(pytorch_file):
    try:
        model = torch.jit.load(pytorch_file)
    except:
        sys.exit('Error: Model file is not Torchscript.\n')


def convert(pytorch_file, input_tensor, caffe_model_path, dump_level=-1, param=None):
    check_model_file(pytorch_file)

    model = Model(pytorch_file, param, input_tensor)
    model.parse()
    model.convert()
    model.save(caffe_model_path)

    if dump_level >= 0:
        model.dump(pytorch_file, param['model_name'], input_tensor, dump_level)

    if dump_level == 3:
        dump_caffe_model(caffe_model_path, input_tensor, param['input_file'])

    compare('pytorch', model, caffe_model_path, input_tensor, param.get('compare', -1))

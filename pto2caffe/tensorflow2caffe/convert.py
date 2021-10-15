import os
import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import meta_graph_pb2 #.MetaGraphDef

from tensorflow2caffe.model import Model
from caffe_dump import dump_caffe_model
from compare import compare

def convert(pb_file, input_tensor, caffe_model_name, caffe_model_path, dump_level=-1, param=None):

    model = Model(pb_file, param)
    model.parse()
    model.convert()
    model.save(caffe_model_name, caffe_model_path)

    if dump_level >= 0:
        model.dump(model_byte, caffe_model_name, input_tensor, dump_level)

    if dump_level == 3:
        dump_caffe_model(caffe_model_name, caffe_model_path, input_tensor, param['input_file'])

    compare('tensorflow', pb_file, caffe_model_name, caffe_model_path, input_tensor, param.get('compare', -1))

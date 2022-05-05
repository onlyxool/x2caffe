import os
import tensorflow as tf
from tensorflow.python.util import compat
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.framework import graph_pb2


from compare import compare
from preprocess import preprocess
from caffe_dump import dump_caffe_model
from tensorflow2caffe.model import Model


def convert(pb_file, input_tensor, caffe_model_path, dump_level=-1, param=None):
    with tf.io.gfile.GFile(pb_file, 'rb') as f:
        data = compat.as_bytes(f.read())
        if os.path.basename(pb_file) == 'saved_model.pb':
            saved_model = saved_model_pb2.SavedModel()
            saved_model.ParseFromString(data)
            graph_def = saved_model.meta_graphs[0].graph_def
            platform = 'SavedModel'
        else:
            graph_def = graph_pb2.GraphDef()
            graph_def.ParseFromString(data)
            platform = 'FrozenModel'

    model = Model(pb_file, graph_def, param)
    model.parse()
    model.convert()
    model.save(caffe_model_path)

    input_tensor = preprocess(input_tensor, param)

    if dump_level >= 0:
        model.dump(graph, param['model_name'], input_tensor, dump_level)

    if dump_level == 3:
        dump_caffe_model(caffe_model_path, input_tensor, param['input_file'])

    compare(platform, model, caffe_model_path, input_tensor, param.get('compare', -1))

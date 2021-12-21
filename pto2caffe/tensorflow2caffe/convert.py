import tensorflow as tf

from tensorflow2caffe.model import Model
from caffe_dump import dump_caffe_model
from compare import compare

def convert(pb_file, input_tensor, caffe_model_name, caffe_model_path, dump_level=-1, param=None):
    with tf.io.gfile.GFile(pb_file, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    model = Model(graph_def, param)
    model.parse()
    model.convert()
    model.save(caffe_model_name, caffe_model_path)

    if dump_level >= 0:
        model.dump(graph, caffe_model_name, input_tensor, dump_level)

    if dump_level == 3:
        dump_caffe_model(caffe_model_name, caffe_model_path, input_tensor, param['input_file'])

    compare('tensorflow', model, caffe_model_name, caffe_model_path, input_tensor, param.get('compare', -1))

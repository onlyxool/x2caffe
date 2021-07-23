import tflite
from tflite2caffe.model import Model

def convert(tf_file, input_tensor, caffe_model_name, caffe_model_path, dump_level=-1, param=None):
    with open(tf_file, 'rb') as f:
        buf = f.read()
        tfmodel = tflite.Model.GetRootAsModel(buf, 0)
    assert(tfmodel.Version() == 3)

    model = Model(tfmodel, param)
    model.parse()
    model.convert()
    model.save(caffe_model_name, caffe_model_path)
'''
def dump(tf_file, input_tensor, dump_level=-1, param=None):
    dump = Dump(onnx_file, caffe_name, input_tensor, param, dump_level)
    from progress_bar import ProgressBar
    i = 0
    progressBar = ProgressBar(len(graph.nodes), 0, "ONNX dump running")
    for id, node in enumerate(graph.nodes):
        dump.layer(node)
        progressBar.setValue(i)
        i += 1
    progressBar.onCancel()
    rnet1 = ONNXModel(onnx_file)
    out = rnet1.forward(input_tensor)
    out_np = np.array(out[0])
    dump.gen_dump_file(out_np, node.name)
'''

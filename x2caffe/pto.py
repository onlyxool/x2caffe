import os
import sys

envroot = os.environ.get('MCHOME', os.environ['PWD'])

exts = ['pb', 'tflite', 'onnx', 'pt']
ext2platform = {
    'onnx': 'onnx',
    'pt': 'pytorch',
    'pb':'tensorflow',
    'tflite': 'tflite',
    'caffemodel': 'caffe'
}


def search_model(path, exts):
    if os.path.isfile(path):
        if path.split('.')[-1].lower() in exts:
            return [os.path.basename(path)], [path]
        else:
            return [], []

    models, models_path = list(), list()
    current_folder = os.walk(path)
    for path, dir_list, file_list in current_folder:
        for file_name in file_list:
            if file_name.split('.')[-1].lower() in exts:
                file_abs = path+'/'+file_name
                models.append(file_name)
                models_path.append(file_abs)

    return models, models_path


def check_op_onnx(model_path, op):
    import onnx
    onnx_model = onnx.load(model_path)
    for node in onnx_model.graph.node:
        if node.op_type == op: 
            return True

    return False


def check_op_tensorflow(model_path, op):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(3)
    import tensorflow as tf
    from tensorflow.python.util import compat
    from tensorflow.core.framework import graph_pb2
    from tensorflow.core.protobuf import saved_model_pb2
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    try:
        if os.path.basename(model_path) == 'saved_model.pb':
            tf_graph = load_saved_model(model_path)
        else:
            tf_graph = load_frozen_model(model_path)
        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(tf_graph, name='')
    except:
        return False

    for index, operation in enumerate(graph.get_operations()):
        if operation.type == op: 
            return True

    return False


def load_saved_model(pb_file):
    modelpath = pb_file.split(os.path.basename(pb_file))[0]

    with tf.io.gfile.GFile(pb_file, 'rb') as f:
        saved_model = saved_model_pb2.SavedModel()
        saved_model.ParseFromString(compat.as_bytes(f.read()))
        meta_graphs = saved_model.meta_graphs

    if len(meta_graphs) > 1:
        tags = meta_graphs[0].meta_info_def.tags[0]
    else:
        tags = None

    imported = tf.saved_model.load(modelpath, tags=tags)

    all_sigs = imported.signatures.keys()
    valid_sigs = [s for s in all_sigs if not s.startswith("_")]
    concrete_func = imported.signatures[valid_sigs[0]]

    frozen_func = convert_variables_to_constants_v2(concrete_func, lower_control_flow=True, aggressive_inlining=True)
    graph_def = frozen_func.graph.as_graph_def(add_shapes=True)

    return graph_def


def load_frozen_model(pb_file):
    with tf.io.gfile.GFile(pb_file, 'rb') as f:
        data = compat.as_bytes(f.read())
        graph_def = graph_pb2.GraphDef()
        graph_def.ParseFromString(data)

    return graph_def


def check_op_tflite(model_path, op):
    import tflite
    with open(model_path, 'rb') as f:
        model_byte = f.read()
        tfmodel = tflite.Model.GetRootAsModel(model_byte, 0)
        graph = tfmodel.Subgraphs(0)

    for index in range(graph.OperatorsLength()):
        tf_op = graph.Operators(index)
        tf_op_code = tfmodel.OperatorCodes(tf_op.OpcodeIndex())
        tf_op_name = tflite.opcode2name(tf_op_code.BuiltinCode())
        if tf_op_name == op:
            return True

    return False


def isContainOp(model_path, platform, op):
    if op is not None:
        if platform == 'onnx':
            return check_op_onnx(model_path, op)
        elif platform == 'tensorflow':
            return check_op_tensorflow(model_path, op)
        elif platform == 'tflite':
            return check_op_tflite(model_path, op)
        else:
            return True


def convert(model_path, convertor, platform, op):
    if op is not None and not isContainOp(model_path, platform, op):
        return

    appex = ' -compare=1 -crop_h=224 -crop_w=224 ' if platform == 'pytorch' else ' -compare=1 '
    simplifier = '-simplifier=1' if platform == 'onnx' else ''
    command = 'python3 ' + convertor + ' -platform=' + platform + ' -model=' + model_path + appex + simplifier

    ret = os.system(command)
    if ret != 0:
        print('\033[0;31;40mError:\033[0m', command, '\n')
    else:
        print('\033[0;32;40mSuccess:\033[0m', command, '\n')

    return ret


def main():
    # Arg parse
    if sys.argv[0].endswith('pyc'):
        convertor = envroot + 'toolchain/x2caffe/convert.pyc'
    elif sys.argv[0].endswith('py'):
        convertor = 'convert.py'

    if len(sys.argv) >= 2:
        if sys.argv[1] == '-h':
            print('')
        else:
            search_path = sys.argv[1]
    else:
        sys.exit('Input model path.')

    # Search Model
    if len(sys.argv) >= 3 and sys.argv[2] in exts:
        models, models_path = search_model(search_path, [sys.argv[2]])
        ext = sys.argv[2]
    else:
        models, models_path = search_model(search_path, exts)
        ext = 'tvm'

    # Run Function
    success, error, ignore = list(), list(), list()
    for index, model_path in enumerate(models_path):
        platform = 'tvm' if ext == 'tvm' else ext2platform[model_path.split('.')[-1].lower()]  
        ret = convert(model_path, convertor, platform, sys.argv[3] if len(sys.argv) >= 4 else None)
        if ret is not None:
            success.append(models[index]) if ret == 0 else error.append(models[index])
        else:
            ignore.append(models[index])

    print('Total:', len(models_path), 'Success:', len(success), 'Error:', len(error), 'Ignore:', len(ignore))


if __name__ == "__main__":
    sys.exit(main())

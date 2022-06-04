import os
import tensorflow as tf
from tensorflow.python.util import compat
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.framework import graph_pb2

from tensorflow.python.framework import graph_util
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.compat.v1.graph_util import extract_sub_graph

from compare import compare
from preprocess import preprocess
from tensorflow2caffe.model import Model


def node_name(name):
    """Get node name without io#."""
    pos = name.find(":")
    if pos >= 0:
        return name[:pos]
    return name


def inputs_without_resource(sess, input_names):
    try:
        new_input_names = []
        for n in input_names:
            t = sess.graph.get_tensor_by_name(n)
            if t.dtype != tf.dtypes.resource:
                new_input_names.append(n)
        input_names = new_input_names
    except:  # pylint: disable=bare-except
        pass
    return input_names


def tf_optimize_grappler(input_names, output_names, graph_def):
    from tensorflow.core.protobuf import meta_graph_pb2 as meta_graph_pb2, config_pb2, rewriter_config_pb2
    from tensorflow.python.grappler import tf_optimizer as tf_opt

    config = config_pb2.ConfigProto()
    rewrite_options = config.graph_options.rewrite_options
    config.graph_options.infer_shapes = True
    # TODO: if we turn on pruning, grappler removes some identities that the tf-1.x lstm rewriter
    # depends on so for now don't turn this on, constfold is always enabled now.
    rewrite_options.optimizers[:] = [
        # 'pruning', 'constfold', 'arithmetic', 'dependency', 'function',
        'constfold', 'function'
    ]

#if LooseVersion(tf.__version__) >= "2.5":
    # This flag disables folding QDQ nodes around constants in the network (eg: around conv/FC weights)
    rewrite_options.experimental_disable_folding_quantization_emulation = True

    meta_graph = tf.compat.v1.train.export_meta_graph(graph_def=graph_def)
    fetch_collection = meta_graph_pb2.CollectionDef()
    for t in input_names + output_names:
        fetch_collection.node_list.value.append(t)
    meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)
    graph_def = tf_opt.OptimizeGraph(config, meta_graph)
    return graph_def


def tf_optimize(input_names, output_names, graph_def):
    """Extract inference subgraph and optimize graph."""
    assert isinstance(input_names, list)
    assert isinstance(output_names, list)

    # TODO: is this needed ?
    needed_names = [node_name(i) for i in input_names] + \
                   [node_name(i) for i in output_names]
    graph_def = extract_sub_graph(graph_def, needed_names)

    graph_def = tf_optimize_grappler(input_names, output_names, graph_def)

    return graph_def


def search_io(graph_def): #TODO: search all outputs
    inputs = []
    outputs = []
    for node in graph_def.node:
        if node.op == 'Placeholder':
            inputs.append(node.name)
    outputs.append(graph_def.node[len(graph_def.node)-1].name)

    return inputs, outputs

def convert(pb_file, input_tensor, caffe_model_path, dump_level=-1, param=None):
    if os.path.basename(pb_file) == 'saved_model.pb':
        modelpath = pb_file.split(os.path.basename(pb_file))[0]
        imported = tf.saved_model.load(modelpath, tags=None)

        all_sigs = imported.signatures.keys()
        valid_sigs = [s for s in all_sigs if not s.startswith("_")]

        concrete_func = imported.signatures[valid_sigs[0]]

        inputs = [tensor.name for tensor in concrete_func.inputs if tensor.dtype != tf.dtypes.resource]
        outputs = [tensor.name for tensor in concrete_func.outputs if tensor.dtype != tf.dtypes.resource]

        frozen_func = convert_variables_to_constants_v2(concrete_func, lower_control_flow=False, aggressive_inlining=True)
        graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
        platform = 'FrozenModel'
    else:
        with tf.io.gfile.GFile(pb_file, 'rb') as f:
            data = compat.as_bytes(f.read())
            graph_def = graph_pb2.GraphDef()
            graph_def.ParseFromString(data)
            inputs, outputs = search_io(graph_def)
            platform = 'FrozenModel'

    if True:
        with tf.Graph().as_default() as tf_graph:
            with tf.compat.v1.Session(graph=tf_graph) as sess:
                tf.import_graph_def(graph_def, name='')
                inputs = inputs_without_resource(sess, inputs)
                graph_def = tf_optimize(inputs, outputs, graph_def)


    model = Model(pb_file, graph_def, param)
    model.parse()
    model.convert()
    model.save(caffe_model_path)

    input_tensor = preprocess(input_tensor, param)

    compare(platform, model, caffe_model_path, input_tensor, param.get('compare', -1))

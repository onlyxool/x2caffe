import os
import tensorflow as tf
from tensorflow.python.util import compat
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from compare import compare
from preprocess import gen_input_tensor
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


def _remove_non_variable_resources_from_captures(concrete_func):
    """
    Removes all non-variable resources (such as tables) from a function's captured inputs to prevent tf from
    raising a 'cannot convert dtype resource to numpy' error while freezing the graph.
    """
    # pylint: disable=protected-access
    resource_id_to_placeholder = {}
    placeholder_to_resource = {}
    graph_captures_copy = None
    func_captures_copy = None
    if hasattr(concrete_func.graph, '_captures') and hasattr(concrete_func, '_captured_inputs'):
        graph_captures_copy = concrete_func.graph._captures.copy()
        func_captures_copy = concrete_func._captured_inputs.copy()
        variable_handles = {id(v.handle) for v in concrete_func.graph.variables}
        for k, v in list(concrete_func.graph._captures.items()):
            val_tensor, name_tensor = v
            if val_tensor.dtype == tf.resource and id(val_tensor) not in variable_handles:
                resource_id_to_placeholder[id(val_tensor)] = name_tensor.name.split(':')[0]
                placeholder_to_resource[name_tensor.name.split(':')[0]] = val_tensor
                del concrete_func.graph._captures[k]
                for i in reversed(range(len(concrete_func._captured_inputs))):
                    if concrete_func._captured_inputs[i] is val_tensor:
                        concrete_func._captured_inputs.pop(i)
            elif val_tensor.dtype != tf.resource:
                npval = val_tensor.numpy()
                if not hasattr(npval, 'dtype'):
                    # Hack around a TF bug until PR is merged: https://github.com/tensorflow/tensorflow/pull/45610
                    arr = np.array(npval)
                    val_tensor.numpy = lambda arr=arr: arr

    return resource_id_to_placeholder, placeholder_to_resource, graph_captures_copy, func_captures_copy


def tf_optimize_grappler(input_names, output_names, graph_def):
    from tensorflow.core.protobuf import meta_graph_pb2 as meta_graph_pb2, config_pb2, rewriter_config_pb2
    from tensorflow.python.grappler import tf_optimizer as tf_opt

    config = config_pb2.ConfigProto()
    rewrite_options = config.graph_options.rewrite_options
    config.graph_options.infer_shapes = True
    # TODO: if we turn on pruning, grappler removes some identities that the tf-1.x lstm rewriter
    # depends on so for now don't turn this on, constfold is always enabled now.
    rewrite_options.optimizers[:] = [
         'pruning', 'constfold', 'arithmetic', 'dependency', 'function'
#        'constfold', 'function'
    ]

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


def shape_inference(graph, inputs_name, param):
    if param['input_shape'] is not None:
        if len(inputs_name) > 1:
            for index, input_name in enumerate(inputs_name):
                graph.get_tensor_by_name(input_name).set_shape(param['input_shape'][index])
        else:
            graph.get_tensor_by_name(inputs_name[0]).set_shape(param['input_shape'])

        with tf.Graph().as_default() as inferred_graph:
            tf.import_graph_def(graph.as_graph_def(add_shapes=True), name="")

        return inferred_graph.as_graph_def(add_shapes=True)
    else:
        return graph.as_graph_def(add_shapes=False)


def is_function(g):
    return 'tensorflow.python.framework.func_graph.FuncGraph' in str(type(g))


def convert(pb_file, caffe_model_path, param=None):
    if os.path.basename(pb_file) == 'saved_model.pb':
        # SavedModel
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

        removed_resource_to_placeholder, placeholder_to_resource, graph_captures_copy, func_captures_copy = \
            _remove_non_variable_resources_from_captures(concrete_func)

        inputs = [tensor.name for tensor in concrete_func.inputs if tensor.dtype != tf.dtypes.resource]
        outputs = [tensor.name for tensor in concrete_func.outputs if tensor.dtype != tf.dtypes.resource]

        frozen_func = convert_variables_to_constants_v2(concrete_func, lower_control_flow=True, aggressive_inlining=True)
        graph_def = frozen_func.graph.as_graph_def(add_shapes=True)

#        print('Is Function:', frozen_func.graph.building_function, ' has been finalized:', frozen_func.graph.finalized)
#        print(' Greph Version:', frozen_func.graph.graph_def_versions, ' Version:', frozen_func.graph.version)
#        print(str(type(graph_def))) #'tensorflow.python.framework.func_graph.FuncGraph'
    else:
        # FrozenModel
        frozen_func = None
        with tf.io.gfile.GFile(pb_file, 'rb') as f:
            data = compat.as_bytes(f.read())
            graph_def = graph_pb2.GraphDef()
            graph_def.ParseFromString(data)
            inputs, outputs = search_io(graph_def)

    # Tensorflow Graph Optimize
    with tf.Graph().as_default() as graph:
        with tf.compat.v1.Session(graph=graph) as sess:
            tf.import_graph_def(graph_def, name='')
            for index, input_name in enumerate(inputs):
                if input_name.find(':') == -1:
                    inputs[index] = input_name+':0'

            graph_def = tf_optimize(inputs, outputs, graph_def)

            # Shape Inference
#if str(type(graph)) in 'tensorflow.python.framework.func_graph.FuncGraph':
            graph_def = shape_inference(graph, inputs, param)


    model = Model(frozen_func, graph_def, param)
    model.parse()
    model.convert()
    caffe_net = model.save(caffe_model_path)

    inputs_tensor = list()
    for index, input_name in enumerate(model.inputs):
        inputs_tensor.append(gen_input_tensor(param, model.inputs_shape[index], model.inputs_dtype[index], None))

    compare(model, caffe_net, inputs_tensor, param.get('compare', -1))

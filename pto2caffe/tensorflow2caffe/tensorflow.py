import os
import tensorflow as tf
from tensorflow import keras


def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


def get_output_frozenmodel(model, input_tensor, blob_name):
    if blob_name.find('split') >= 0:
        return None

    # Wrap frozen graph to ConcreteFunctions
    frozen_func = wrap_frozen_graph(graph_def=model.graph,
                                    inputs=[model.inputs[0]],
                                    outputs=[blob_name],
                                    print_graph=True)

    output = frozen_func(image_input=tf.constant(list(input_tensor)))

    return output[0].numpy().transpose(0, 3, 1, 2) if len(output[0].numpy().shape) == 4 else output[0].numpy()



def get_output_savedmodel(model, input_tensor, blob_name):
    if blob_name.find('split') >= 0:
        return None

    if model.model is None:
        modelpath = model.model_file.split(os.path.basename(model.model_file))[0]
        model.model = tf.saved_model.load(modelpath)

    input_name = model.model.signatures['default'].inputs[0].name

    pruned = model.model.prune(input_name, blob_name)
    output = pruned(tf.constant(input_tensor)).numpy()

    return output.transpose(0, 3, 1, 2) if len(output.shape) == 4 else output

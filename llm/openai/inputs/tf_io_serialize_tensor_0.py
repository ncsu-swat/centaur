
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_io_serialize_tensor_inputs():
    list_of_inputs = []

    input_dict = {
        "tensor": tf.constant(1),
        "name": "scalar_tensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": tf.constant([1, 2, 3]),
        "name": "vector_tensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": tf.constant([[1, 2], [3, 4]]),
        "name": "matrix_tensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        "name": "3d_tensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": tf.constant([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]),
        "name": "4d_tensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.io.serialize_tensor"] = tf_io_serialize_tensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.io.serialize_tensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.serialize_tensor'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.io.serialize_tensor', generated_inputs['tf.io.serialize_tensor'], lib="tf", suffix=0)

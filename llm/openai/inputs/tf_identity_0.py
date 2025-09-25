
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_identity_inputs():
    list_of_inputs = []
    input_dict = {}
    generated_inputs = {}

    input_dict = {
        "input": tf.constant([1, 2, 3]),
        "name": "identity_tensor_1d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": tf.constant([[1, 2], [3, 4]]),
        "name": "identity_tensor_2d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        "name": "identity_tensor_3d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": tf.constant([-1, -2, -3]),
        "name": "identity_tensor_negative"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": tf.constant([0.1, 0.2, 0.3]),
        "name": "identity_tensor_float"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": tf.constant([1, 0, 1, 0]),
        "name": "identity_tensor_bool"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": tf.constant([1 + 1j, 2 + 2j]),
        "name": "identity_tensor_complex"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": tf.constant([1, 2, 3], dtype=tf.int64),
        "name": "identity_tensor_int64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": tf.constant([1.0, 2.0, 3.0], dtype=tf.float64),
        "name": "identity_tensor_float64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": tf.constant([True, False, True], dtype=tf.bool),
        "name": "identity_tensor_bool_type"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.identity"] = tf_identity_inputs()


def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.identity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.identity'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.identity', generated_inputs['tf.identity'], lib="tf", suffix=0)

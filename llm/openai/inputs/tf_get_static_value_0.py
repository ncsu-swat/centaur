
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_get_static_value_inputs():
    list_of_inputs = []

    input_dict = {
        "tensor": tf.constant(10),
        "partial": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": tf.constant([1, 2, 3]),
        "partial": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": tf.constant([[1, 2], [3, 4]]),
        "partial": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        "partial": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": tf.constant(-5),
        "partial": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": tf.constant([-1, -2, -3]),
        "partial": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": tf.constant(0.5),
        "partial": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": tf.constant([1.0, 2.5, -3.2]),
        "partial": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": tf.constant(10),
        "partial": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": tf.constant(-10),
        "partial": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.get_static_value"] = tf_get_static_value_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.get_static_value' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.get_static_value'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.get_static_value', generated_inputs['tf.get_static_value'], lib="tf", suffix=0)

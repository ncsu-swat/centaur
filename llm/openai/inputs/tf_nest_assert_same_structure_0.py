
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_nest_assert_same_structure_inputs():
    list_of_inputs = []

    input1 = {
        "nest1": [1, 2, 3],
        "nest2": [4, 5, 6],
        "check_types": True,
        "expand_composites": False
    }
    list_of_inputs.append(copy.deepcopy(input1))

    input2 = {
        "nest1": [[1, 2], [3, 4]],
        "nest2": [[5, 6], [7, 8]],
        "check_types": True,
        "expand_composites": False
    }
    list_of_inputs.append(copy.deepcopy(input2))

    input3 = {
        "nest1": ([1, 2], 3),
        "nest2": ([4, 5], 6),
        "check_types": True,
        "expand_composites": False
    }
    list_of_inputs.append(copy.deepcopy(input3))

    input4 = {
        "nest1": (1, 2, 3),
        "nest2": (4, 5, 6),
        "check_types": True,
        "expand_composites": False
    }
    list_of_inputs.append(copy.deepcopy(input4))

    input5 = {
        "nest1": [np.array([1, 2]), np.array([3, 4])],
        "nest2": [np.array([5, 6]), np.array([7, 8])],
        "check_types": True,
        "expand_composites": False
    }
    list_of_inputs.append(copy.deepcopy(input5))

    input6 = {
        "nest1": [1, [2, 3]],
        "nest2": [4, [5, 6]],
        "check_types": True,
        "expand_composites": False
    }
    list_of_inputs.append(copy.deepcopy(input6))
    
    input7 = {
        "nest1": (1, [2, 3], 4),
        "nest2": (5, [6, 7], 8),
        "check_types": True,
        "expand_composites": False
    }
    list_of_inputs.append(copy.deepcopy(input7))

    return list_of_inputs

generated_inputs["tf.nest.assert_same_structure"] = tf_nest_assert_same_structure_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nest.assert_same_structure' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nest.assert_same_structure'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nest.assert_same_structure', generated_inputs['tf.nest.assert_same_structure'], lib="tf", suffix=0)

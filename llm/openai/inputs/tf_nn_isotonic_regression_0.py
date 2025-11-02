
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_isotonic_regression_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input_1 = np.array([[3, 1, 2], [1, 3, 4]], dtype=np.float32)
    input_dict = {
        "inputs": tf.constant(input_1),
        "decreasing": True,
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input_2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    input_dict = {
        "inputs": tf.constant(input_2),
        "decreasing": True,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input_3 = np.array([1, 2, 3, 4], dtype=np.float32)
    input_dict = {
        "inputs": tf.constant(input_3),
        "decreasing": False,
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input_4 = np.array([[5, 4, 3], [2, 1, 0]], dtype=np.float32)
    input_dict = {
        "inputs": tf.constant(input_4),
        "decreasing": False,
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input_5 = np.array([[-1, -2, -3], [1, 2, 3]], dtype=np.float32)
    input_dict = {
        "inputs": tf.constant(input_5),
        "decreasing": True,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input_6 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    input_dict = {
        "inputs": tf.constant(input_6),
        "decreasing": True,
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input_7 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    input_dict = {
        "inputs": tf.constant(input_7),
        "decreasing": False,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input_8 = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype=np.float32)
    input_dict = {
        "inputs": tf.constant(input_8),
        "decreasing": True,
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input_9 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    input_dict = {
        "inputs": tf.constant(input_9),
        "decreasing": True,
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input_10 = np.array([[5, 4, 3], [2, 1, 0]], dtype=np.float32)
    input_dict = {
        "inputs": tf.constant(input_10),
        "decreasing": False,
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.isotonic_regression"] = tf_isotonic_regression_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.isotonic_regression' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.isotonic_regression'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.isotonic_regression', generated_inputs['tf.nn.isotonic_regression'], lib="tf", suffix=0)

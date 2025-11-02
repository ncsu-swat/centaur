
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_round_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensor
    x = np.array([1.4, 2.7, 3.1, 4.5], dtype=np.float32)
    input_dict = {
        "name": "round1",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: float64 tensor
    x = np.array([1.4, 2.7, 3.1, 4.5], dtype=np.float64)
    input_dict = {
        "name": "round2",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: int32 tensor
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    input_dict = {
        "name": "round3",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: int64 tensor
    x = np.array([1, 2, 3, 4], dtype=np.int64)
    input_dict = {
        "name": "round4",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: float32 tensor with negative values
    x = np.array([-1.4, -2.7, -3.1, -4.5], dtype=np.float32)
    input_dict = {
        "name": "round5",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: float64 tensor with negative values
    x = np.array([-1.4, -2.7, -3.1, -4.5], dtype=np.float64)
    input_dict = {
        "name": "round6",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: int32 tensor with negative values
    x = np.array([-1, -2, -3, -4], dtype=np.int32)
    input_dict = {
        "name": "round7",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: int64 tensor with negative values
    x = np.array([-1, -2, -3, -4], dtype=np.int64)
    input_dict = {
        "name": "round8",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: float32 tensor with mixed values
    x = np.array([1.5, -2.5, 3.5, -4.5], dtype=np.float32)
    input_dict = {
        "name": "round9",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: float64 tensor with mixed values
    x = np.array([1.5, -2.5, 3.5, -4.5], dtype=np.float64)
    input_dict = {
        "name": "round10",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Round"] = generate_round_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Round' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Round'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Round', generated_inputs['tf.raw_ops.Round'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_gather_inputs():
    list_of_inputs = []
    
    # Input 1: Scalar indices
    params = np.array([[1, 2, 3], [4, 5, 6]])
    indices = np.array(1, dtype=np.int32)
    validate_indices = True
    name = "test1"
    
    input_dict = {
        "params": params,
        "indices": indices,
        "validate_indices": validate_indices,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Vector indices
    params = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    indices = np.array([0, 1, 2], dtype=np.int32)
    validate_indices = True
    name = "test2"
    
    input_dict = {
        "params": params,
        "indices": indices,
        "validate_indices": validate_indices,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Higher rank indices (2D)
    params = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    indices = np.array([[0, 1], [1, 0]], dtype=np.int32)
    validate_indices = True
    name = "test3"
    
    input_dict = {
        "params": params,
        "indices": indices,
        "validate_indices": validate_indices,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Empty indices
    params = np.array([[1, 2], [3, 4], [5, 6]])
    indices = np.array([], dtype=np.int32)
    validate_indices = True
    name = "test4"
    
    input_dict = {
        "params": params,
        "indices": indices,
        "validate_indices": validate_indices,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Single element indices
    params = np.array([[1, 2], [3, 4], [5, 6]])
    indices = np.array([0], dtype=np.int32)
    validate_indices = True
    name = "test5"
    
    input_dict = {
        "params": params,
        "indices": indices,
        "validate_indices": validate_indices,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Large indices (with repeated values)
    params = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    indices = np.array([2, 2, 0, 1], dtype=np.int32)
    validate_indices = True
    name = "test6"
    
    input_dict = {
        "params": params,
        "indices": indices,
        "validate_indices": validate_indices,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 1D indices with single element
    params = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    indices = np.array([2], dtype=np.int32)
    validate_indices = True
    name = "test7"
    
    input_dict = {
        "params": params,
        "indices": indices,
        "validate_indices": validate_indices,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Multi-dimensional indices with shape (2, 2)
    params = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    indices = np.array([[1, 0], [0, 1]], dtype=np.int32)
    validate_indices = True
    name = "test8"
    
    input_dict = {
        "params": params,
        "indices": indices,
        "validate_indices": validate_indices,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Gather"] = tf_raw_ops_gather_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Gather' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Gather'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Gather', generated_inputs['tf.raw_ops.Gather'], lib="tf", suffix=0)

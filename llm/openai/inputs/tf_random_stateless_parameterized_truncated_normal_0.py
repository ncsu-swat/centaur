
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_inputs():
    list_of_inputs = []
    
    # Input 1
    shape = np.array([2, 3], dtype=np.int32)
    seed = np.array([1, 2], dtype=np.int32)
    means = np.array(0.0, dtype=np.float32)
    stddevs = np.array(1.0, dtype=np.float32)
    minvals = np.array(-2.0, dtype=np.float32)
    maxvals = np.array(2.0, dtype=np.float32)
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "means": means,
        "stddevs": stddevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "name": "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    shape = np.array([5], dtype=np.int32)
    seed = np.array([7, 17], dtype=np.int32)
    means = np.array([0.0], dtype=np.float32)
    stddevs = np.array([1.0], dtype=np.float32)
    minvals = np.array([-2.0], dtype=np.float32)
    maxvals = np.array([2.0], dtype=np.float32)
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "means": means,
        "stddevs": stddevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "name": "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    shape = np.array([3, 4], dtype=np.int32)
    seed = np.array([10, 20], dtype=np.int32)
    means = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    stddevs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    minvals = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    maxvals = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "means": means,
        "stddevs": stddevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "name": "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    shape = np.array([1, 2, 3], dtype=np.int32)
    seed = np.array([100, 200], dtype=np.int32)
    means = np.array(0.5, dtype=np.float32)
    stddevs = np.array(0.75, dtype=np.float32)
    minvals = np.array(-1.5, dtype=np.float32)
    maxvals = np.array(1.5, dtype=np.float32)
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "means": means,
        "stddevs": stddevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "name": "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    shape = np.array([2, 2, 2], dtype=np.int32)
    seed = np.array([77, 177], dtype=np.int32)
    means = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    stddevs = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    minvals = np.array([-2.0, -2.0], dtype=np.float32)
    maxvals = np.array([2.0, 2.0], dtype=np.float32)
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "means": means,
        "stddevs": stddevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "name": "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    shape = np.array([1, 1], dtype=np.int32)
    seed = np.array([99, 100], dtype=np.int32)
    means = np.array([0.0], dtype=np.float32)
    stddevs = np.array([1.0], dtype=np.float32)
    minvals = np.array([-2.0], dtype=np.float32)
    maxvals = np.array([2.0], dtype=np.float32)
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "means": means,
        "stddevs": stddevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "name": "test6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    shape = np.array([4], dtype=np.int32)
    seed = np.array([1, 2], dtype=np.int32)
    means = np.array(0.5, dtype=np.float32)
    stddevs = np.array(0.25, dtype=np.float32)
    minvals = np.array(-1.0, dtype=np.float32)
    maxvals = np.array(1.0, dtype=np.float32)
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "means": means,
        "stddevs": stddevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "name": "test7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    shape = np.array([10, 5], dtype=np.int32)
    seed = np.array([4, 5], dtype=np.int32)
    means = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    stddevs = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    minvals = np.array([-2.0], dtype=np.float32)
    maxvals = np.array([2.0], dtype=np.float32)
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "means": means,
        "stddevs": stddevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "name": "test8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    shape = np.array([2, 3, 4], dtype=np.int32)
    seed = np.array([100, 200], dtype=np.int32)
    means = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=np.float32)
    stddevs = np.array([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]], dtype=np.float32)
    minvals = np.array([-2.0, -2.0, -2.0], dtype=np.float32)
    maxvals = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "means": means,
        "stddevs": stddevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "name": "test9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    shape = np.array([2, 3], dtype=np.int32)
    seed = np.array([7, 17], dtype=np.int32)
    means = np.array(0.0, dtype=np.float32)
    stddevs = np.array(1.0, dtype=np.float32)
    minvals = np.array(-2.0, dtype=np.float32)
    maxvals = np.array(2.0, dtype=np.float32)
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "means": means,
        "stddevs": stddevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "name": "test10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.random.stateless_parameterized_truncated_normal"] = generate_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random.stateless_parameterized_truncated_normal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_parameterized_truncated_normal'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.random.stateless_parameterized_truncated_normal', generated_inputs['tf.random.stateless_parameterized_truncated_normal'], lib="tf", suffix=0)

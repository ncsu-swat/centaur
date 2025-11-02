
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_stateless_gamma_inputs():
    list_of_inputs = []
    
    # Input 1: Basic case with scalar alpha and beta
    shape = np.array([5], dtype=np.int32)
    seed = np.array([12, 34], dtype=np.int32)
    alpha = np.array([0.5], dtype=np.float32)
    beta = np.array([1.5], dtype=np.float32)
    dtype = np.float32
    name = "test"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Shape with two dimensions, alpha and beta are vectors
    shape = np.array([3, 4], dtype=np.int32)
    seed = np.array([56, 78], dtype=np.int32)
    alpha = np.array([1.0, 2.0], dtype=np.float32)
    beta = np.array([0.5, 1.5], dtype=np.float32)
    dtype = np.float32
    name = "test2"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Multi-dimensional shape, scalar alpha and beta
    shape = np.array([2, 3, 4], dtype=np.int32)
    seed = np.array([11, 22], dtype=np.int32)
    alpha = np.array([0.75], dtype=np.float32)
    beta = np.array([1.25], dtype=np.float32)
    dtype = np.float64
    name = "test3"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: With negative alpha values
    shape = np.array([2, 3], dtype=np.int32)
    seed = np.array([99, 100], dtype=np.int32)
    alpha = np.array([-1.5, -2.5], dtype=np.float32)
    beta = np.array([1.5, 2.5], dtype=np.float32)
    dtype = np.float32
    name = "test4"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Single dimension with broadcasting dimensions
    shape = np.array([10], dtype=np.int32)
    seed = np.array([45, 67], dtype=np.int32)
    alpha = np.array([[1.5], [2.5]], dtype=np.float32)
    beta = np.array([[0.5, 1.5]], dtype=np.float32)
    dtype = np.float32
    name = "test5"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Large shape with scalar alpha and beta
    shape = np.array([100], dtype=np.int32)
    seed = np.array([89, 90], dtype=np.int32)
    alpha = np.array([1.0], dtype=np.float32)
    beta = np.array([1.0], dtype=np.float32)
    dtype = np.float64
    name = "test6"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: With float64 dtype
    shape = np.array([5, 5], dtype=np.int32)
    seed = np.array([123, 456], dtype=np.int64)
    alpha = np.array([0.25, 0.5], dtype=np.float64)
    beta = np.array([0.75, 1.5], dtype=np.float64)
    dtype = np.float64
    name = "test7"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Mixed dimensions with scalar alpha and beta
    shape = np.array([2, 3, 4], dtype=np.int32)
    seed = np.array([789, 123], dtype=np.int32)
    alpha = np.array([0.5], dtype=np.float32)
    beta = np.array([1.0], dtype=np.float32)
    dtype = np.float32
    name = "test8"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Large number of dimensions
    shape = np.array([3, 4, 5, 6], dtype=np.int32)
    seed = np.array([1234, 5678], dtype=np.int32)
    alpha = np.array([0.1], dtype=np.float32)
    beta = np.array([0.2], dtype=np.float32)
    dtype = np.float32
    name = "test9"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Different seed values, multiple alpha and beta
    shape = np.array([3, 3], dtype=np.int32)
    seed = np.array([999, 888], dtype=np.int32)
    alpha = np.array([1.0, 2.0], dtype=np.float32)
    beta = np.array([0.5, 1.5], dtype=np.float32)
    dtype = np.float64
    name = "test10"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.random.stateless.gamma"] = tf_random_stateless_gamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random.stateless_gamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_gamma'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.random.stateless_gamma', generated_inputs['tf.random.stateless_gamma'], lib="tf", suffix=0)

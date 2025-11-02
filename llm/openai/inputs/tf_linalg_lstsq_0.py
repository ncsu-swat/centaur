
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_lstsq_inputs():
    list_of_inputs = []
    
    # Input 1: Basic case with fast=True
    matrix = np.array([[1., 2.], [3., 4.], [5., 6.]], dtype=np.float32)
    rhs = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    l2_regularizer = 0.0
    fast = True
    name = "test1"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "l2_regularizer": l2_regularizer,
        "fast": fast,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: With regularization
    matrix = np.array([[1., 2.], [3., 4.], [5., 6.]], dtype=np.float32)
    rhs = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    l2_regularizer = 0.1
    fast = True
    name = "test2"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "l2_regularizer": l2_regularizer,
        "fast": fast,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: With fast=False
    matrix = np.array([[1., 2.], [3., 4.], [5., 6.]], dtype=np.float32)
    rhs = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    l2_regularizer = 0.0
    fast = False
    name = "test3"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "l2_regularizer": l2_regularizer,
        "fast": fast,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Larger matrix with multiple right-hand sides
    matrix = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]], dtype=np.float32)
    rhs = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
    l2_regularizer = 0.0
    fast = True
    name = "test4"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "l2_regularizer": l2_regularizer,
        "fast": fast,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Different shape matrices
    matrix = np.array([[1., 2.], [3., 4.], [5., 6.]], dtype=np.float32)
    rhs = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]], dtype=np.float32)
    l2_regularizer = 0.0
    fast = True
    name = "test5"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "l2_regularizer": l2_regularizer,
        "fast": fast,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Negative values in matrix
    matrix = np.array([[-1., 2.], [3., -4.]], dtype=np.float32)
    rhs = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    l2_regularizer = 0.0
    fast = True
    name = "test6"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "l2_regularizer": l2_regularizer,
        "fast": fast,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Single row matrix
    matrix = np.array([[1., 2.]], dtype=np.float32)
    rhs = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    l2_regularizer = 0.0
    fast = True
    name = "test7"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "l2_regularizer": l2_regularizer,
        "fast": fast,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Single column matrix with multiple RHS
    matrix = np.array([[1.], [2.], [3.]], dtype=np.float32)
    rhs = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
    l2_regularizer = 0.0
    fast = True
    name = "test8"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "l2_regularizer": l2_regularizer,
        "fast": fast,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Complex matrix with multiple RHS
    matrix = np.array([[1.+1j, 2.+2j], [3.+3j, 4.+4j]], dtype=np.complex64)
    rhs = np.array([[1.+1j, 2.+2j], [3.+3j, 4.+4j]], dtype=np.complex64)
    l2_regularizer = 0.0
    fast = True
    name = "test9"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "l2_regularizer": l2_regularizer,
        "fast": fast,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Large matrix with small RHS
    matrix = np.array([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]], dtype=np.float32)
    rhs = np.array([[1., 2.]], dtype=np.float32)
    l2_regularizer = 0.0
    fast = True
    name = "test10"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "l2_regularizer": l2_regularizer,
        "fast": fast,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.linalg.lstsq"] = tf_linalg_lstsq_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.lstsq' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.lstsq'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.lstsq', generated_inputs['tf.linalg.lstsq'], lib="tf", suffix=0)

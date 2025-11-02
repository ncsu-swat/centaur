
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_solve_inputs():
    list_of_inputs = []
    
    # Input 1, valid - basic case with float32
    matrix = np.array([[2.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    rhs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    adjoint = False
    name = "test"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid - with complex64
    matrix = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    rhs = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    adjoint = True
    name = "test2"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid - with float64
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    rhs = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    adjoint = False
    name = "test3"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid - with complex128
    matrix = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128)
    rhs = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128)
    adjoint = True
    name = "test4"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid - negative values with float32
    matrix = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    rhs = np.array([[5.0, 6.0], [-7.0, 8.0]], dtype=np.float32)
    adjoint = True
    name = "test5"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid - with negative values
    matrix = np.array([[1.0, -2.0], [-3.0, 4.0]], dtype=np.float32)
    rhs = np.array([[5.0, -6.0], [7.0, -8.0]], dtype=np.float32)
    adjoint = True
    name = "test6"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid - single dimension matrices with float32
    matrix = np.array([[1.0]], dtype=np.float32)
    rhs = np.array([[2.0]], dtype=np.float32)
    adjoint = False
    name = "test7"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid - float32 with more complex structure but invertible
    matrix = np.array([[2.0, 1.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    rhs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    adjoint = False
    name = "test8"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid - with float32 but different dimensions
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rhs = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    adjoint = False
    name = "test9"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid - float32 with different dimensions and invertible matrix
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rhs = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    adjoint = True
    name = "test10"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.linalg.solve"] = tf_linalg_solve_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.solve'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.solve', generated_inputs['tf.linalg.solve'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linear_operator_householder_inputs():
    list_of_inputs = []
    
    # Input 1 - Basic Householder reflection with 1D vector
    reflection_axis = np.array([1.0, 0.0])
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = None
    is_square = True
    name = "Householder_1D"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - 2D vector with non-singular flag
    reflection_axis = np.array([0.5, 0.5])
    is_non_singular = True
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "Householder_2D"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - 3D vector with positive definite flag
    reflection_axis = np.array([1.0, 1.0, 1.0])
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = None
    is_square = True
    name = "Householder_3D"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - 4D vector with square flag
    reflection_axis = np.array([1.0, 0.0, 0.0, 1.0])
    is_non_singular = True
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "Householder_4D"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - Batch vector with non-singular flag
    reflection_axis = np.array([[1.0, 0.0], [0.0, 1.0]])
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = None
    is_square = True
    name = "Householder_batch"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - Negative values vector
    reflection_axis = np.array([-1.0, 1.0])
    is_non_singular = True
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "Householder_negative"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - Floating point vector with non-singular flag
    reflection_axis = np.array([0.1, 0.2])
    is_non_singular = True
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "Householder_float"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - Zero vector (not recommended but valid)
    reflection_axis = np.array([0.0, 0.0])
    is_non_singular = False
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "Householder_zero"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - Higher dimensional vector (5D)
    reflection_axis = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = None
    is_square = True
    name = "Householder_5D"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - Complex vector with multiple dimensions
    reflection_axis = np.array([1.0 + 1j, 0.0 + 0j])
    is_non_singular = True
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "Householder_complex"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorHouseholder"] = tf_linalg_linear_operator_householder_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.LinearOperatorHouseholder' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorHouseholder'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.LinearOperatorHouseholder', generated_inputs['tf.linalg.LinearOperatorHouseholder'], lib="tf", suffix=0)

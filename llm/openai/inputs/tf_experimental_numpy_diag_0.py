
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_diag_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    v = np.array([1, 2, 3, 4])
    k = 0
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = 0
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = 1
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = -1
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    v = np.array([1, 2, 3, 4])
    k = 1
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    v = np.array([[1, 2], [3, 4]])
    k = 0
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    v = np.array([1, 2, 3, 4])
    k = -1
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    v = np.array([[1, 2, 3], [4, 5, 6]])
    k = 2
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    v = np.array([[1, 2], [3, 4]])
    k = 0
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = -2
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.diag"] = tf_experimental_numpy_diag_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.diag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.diag'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.diag', generated_inputs['tf.experimental.numpy.diag'], lib="tf", suffix=0)

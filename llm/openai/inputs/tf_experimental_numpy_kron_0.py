
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def kron_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2, valid
    a = np.array([1, 2, 3])
    b = np.array([4, 5])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3, valid
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([7, 8])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4, valid
    a = np.array([[1, 2], [3, 4], [5, 6]])
    b = np.array([[7, 8, 9], [10, 11, 12]])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = np.array([[9, 10], [11, 12]])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6, valid - negative values
    a = np.array([[-1, 2], [3, -4]])
    b = np.array([[5, -6], [-7, 8]])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7, valid - floating point
    a = np.array([1.5, 2.7])
    b = np.array([3.1, 4.2])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8, valid - mixed types
    a = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9, valid - scalar
    a = np.array(5)
    b = np.array(2)
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10, valid - 4D array
    a = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])
    b = np.array([[17, 18], [19, 20]])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.kron"] = kron_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.kron' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.kron'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.kron', generated_inputs['tf.experimental.numpy.kron'], lib="tf", suffix=0)

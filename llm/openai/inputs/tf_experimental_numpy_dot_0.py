
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_dot_inputs():
    list_of_inputs = []
    
    # Input 1: Valid 2D tensors
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Valid 1D tensors
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Valid 3D tensors
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Mixed types (float and int)
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5, 6], [7, 8]])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Negative values
    a = np.array([[-1, -2], [-3, -4]])
    b = np.array([[1, 2], [3, 4]])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Scalar tensors (1D)
    a = np.array([1])
    b = np.array([2])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Different dimensions
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[1, 2], [3, 4], [5, 6]])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Zero values
    a = np.array([[0, 0], [0, 0]])
    b = np.array([[1, 2], [3, 4]])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Floating point tensors
    a = np.array([[1.5, 2.7], [3.1, 4.8]])
    b = np.array([[5.2, 6.3], [7.9, 8.4]])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Large values (int)
    a = np.array([[100, 200], [300, 400]])
    b = np.array([[500, 600], [700, 800]])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.dot"] = tf_dot_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.dot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.dot'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.dot', generated_inputs['tf.experimental.numpy.dot'], lib="tf", suffix=0)

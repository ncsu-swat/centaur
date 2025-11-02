
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_lgamma_inputs():
    list_of_inputs = []
    
    # Input 1: Positive integers
    x = np.array([1, 2, 3, 4], dtype=np.float32)
    input_dict = {"x": x, "name": "positive_integers"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Positive floats
    x = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    input_dict = {"x": x, "name": "positive_floats"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Negative floats (with non-integer values)
    x = np.array([-0.5, -1.5, -2.5], dtype=np.float32)
    input_dict = {"x": x, "name": "negative_floats"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Mixed positive and negative values
    x = np.array([0, 0.5, 1, -1, -2], dtype=np.float64)
    input_dict = {"x": x, "name": "mixed_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Single element array
    x = np.array([3.5], dtype=np.float32)
    input_dict = {"x": x, "name": "single_element"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Negative integers
    x = np.array([-1, -2, -3], dtype=np.float32)
    input_dict = {"x": x, "name": "negative_integers"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Large positive values
    x = np.array([10.5, 15.7, 20.2], dtype=np.float64)
    input_dict = {"x": x, "name": "large_positive"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Small positive values
    x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    input_dict = {"x": x, "name": "small_positive"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Zero and negative values
    x = np.array([0, -0.5, -1.5], dtype=np.float64)
    input_dict = {"x": x, "name": "zero_and_negatives"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Float arrays with different shapes
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input_dict = {"x": x, "name": "multi_dimensional"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.lgamma"] = generate_lgamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.lgamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.lgamma'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.lgamma', generated_inputs['tf.math.lgamma'], lib="tf", suffix=0)

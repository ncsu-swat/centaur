
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_bessel_j1_inputs():
    list_of_inputs = []
    
    # Input 1: scalar tensor
    x = tf.constant(0.5, dtype=tf.float32)
    input_dict = {"x": x, "name": "test1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 1D tensor
    x = tf.constant([0.5, 1., 2., 4.], dtype=tf.float32)
    input_dict = {"x": x, "name": "test2"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 2D tensor
    x = tf.constant([[0.5, 1.], [2., 4.]], dtype=tf.float32)
    input_dict = {"x": x, "name": "test3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: negative values
    x = tf.constant([-0.5, -1., -2., -4.], dtype=tf.float32)
    input_dict = {"x": x, "name": "test4"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: mixed positive and negative values
    x = tf.constant([0.5, -1., 2., -4.], dtype=tf.float32)
    input_dict = {"x": x, "name": "test5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: float64 tensor
    x = tf.constant([0.5, 1., 2., 4.], dtype=tf.float64)
    input_dict = {"x": x, "name": "test6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: scalar tensor with float64
    x = tf.constant(0.5, dtype=tf.float64)
    input_dict = {"x": x, "name": "test7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: zero values
    x = tf.constant([0., 1., 2., 4.], dtype=tf.float32)
    input_dict = {"x": x, "name": "test8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: large values
    x = tf.constant([100., 200., 300., 400.], dtype=tf.float32)
    input_dict = {"x": x, "name": "test9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: small values
    x = tf.constant([0.001, 0.01, 0.1, 0.0001], dtype=tf.float32)
    input_dict = {"x": x, "name": "test10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.special.bessel_j1"] = generate_bessel_j1_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.special.bessel_j1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.special.bessel_j1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.special.bessel_j1', generated_inputs['tf.math.special.bessel_j1'], lib="tf", suffix=0)

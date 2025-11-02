
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_random_hue_inputs():
    list_of_inputs = []
    
    # Input 1
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    max_delta = 0.2
    seed = 42
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    image = np.array([[[0.5, 0.8, 0.3], [0.1, 0.9, 0.7]], [[0.2, 0.6, 0.4], [0.0, 0.5, 0.2]]], dtype=np.float32)
    max_delta = 0.1
    seed = 123
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    image = np.array([[[1.0, 0.0, 0.5], [0.2, 0.8, 0.9]], [[0.7, 0.3, 0.1], [0.6, 0.4, 0.0]]], dtype=np.float32)
    max_delta = 0.4
    seed = 50
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    image = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], dtype=np.float32)
    max_delta = 0.3
    seed = 99
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    image = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [0.1, 0.2, 0.3]]], dtype=np.float32)
    max_delta = 0.0
    seed = 100
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    image = np.array([[[0.5, 0.8, 0.3], [0.1, 0.9, 0.7]], [[0.2, 0.6, 0.4], [0.0, 0.5, 0.2]]], dtype=np.float32)
    max_delta = 0.2
    seed = 200
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    max_delta = 0.5
    seed = 400
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    image = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [0.1, 0.2, 0.3]]], dtype=np.float32)
    max_delta = 0.4
    seed = 500
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    image = np.array([[[0.5, 0.8, 0.3], [0.1, 0.9, 0.7]], [[0.2, 0.6, 0.4], [0.0, 0.5, 0.2]]], dtype=np.float32)
    max_delta = 0.1
    seed = 600
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    image = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], dtype=np.float32)
    max_delta = 0.4
    seed = 700
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.image.random_hue"] = tf_image_random_hue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.random_hue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.random_hue'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.random_hue', generated_inputs['tf.image.random_hue'], lib="tf", suffix=0)

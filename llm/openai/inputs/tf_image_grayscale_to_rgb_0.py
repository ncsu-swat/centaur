
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_grayscale_to_rgb_inputs():
    list_of_inputs = []
    
    # Input 1: Simple 1D grayscale image
    images = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32)
    input_dict = {
        "images": images,
        "name": "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D grayscale image
    images = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
    input_dict = {
        "images": images,
        "name": "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D grayscale image with multiple channels
    images = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32)
    input_dict = {
        "images": images,
        "name": "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 4D grayscale image
    images = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
    input_dict = {
        "images": images,
        "name": "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Negative values in grayscale image
    images = np.array([[[1.0], [-2.0], [3.0]]], dtype=np.float32)
    input_dict = {
        "images": images,
        "name": "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Large values in grayscale image
    images = np.array([[[100.0], [200.0], [300.0]]], dtype=np.float32)
    input_dict = {
        "images": images,
        "name": "test6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Float values in grayscale image
    images = np.array([[[1.5], [2.7], [3.9]]], dtype=np.float32)
    input_dict = {
        "images": images,
        "name": "test7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Mixed values in grayscale image
    images = np.array([[[1.0], [2.5], [3.9]], [[4.2], [5.7], [6.1]]], dtype=np.float32)
    input_dict = {
        "images": images,
        "name": "test8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Single pixel grayscale image
    images = np.array([[[1.0]], [[2.0]]], dtype=np.float32)
    input_dict = {
        "images": images,
        "name": "test9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Multiple grayscale images with different shapes
    images = np.array([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]], dtype=np.float32)
    input_dict = {
        "images": images,
        "name": "test10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.grayscale_to_rgb"] = generate_grayscale_to_rgb_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.grayscale_to_rgb' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.grayscale_to_rgb'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.grayscale_to_rgb', generated_inputs['tf.image.grayscale_to_rgb'], lib="tf", suffix=0)

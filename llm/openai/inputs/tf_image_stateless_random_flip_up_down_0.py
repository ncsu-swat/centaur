
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_flip_up_down_inputs():
    list_of_inputs = []
    
    # Input 1: 3D tensor (height, width, channels)
    image = np.array([[[1], [2]], [[3], [4]]], dtype=np.int32)
    seed = np.array([2, 3], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 4D tensor (batch, height, width, channels)
    image = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.int32)
    seed = np.array([5, 7], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D tensor with different channel count
    image = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    seed = np.array([1, 9], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D tensor with negative values
    image = np.array([[[1, -2], [-3, 4]], [[5, -6], [7, -8]]], dtype=np.int32)
    seed = np.array([10, 15], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 4D tensor with single channel
    image = np.array([[[[1]], [[2]]]], dtype=np.int32)
    seed = np.array([1, 1], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 4D tensor with multiple channels
    image = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int32)
    seed = np.array([100, 200], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 4D tensor with mixed values
    image = np.array([[[[1, -2], [3, 4]], [[-5, 6], [-7, 8]]]], dtype=np.int32)
    seed = np.array([1000, 2000], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 3D tensor with zero values
    image = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]], dtype=np.int32)
    seed = np.array([50, 50], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 4D tensor with large values
    image = np.array([[[[100, 200], [300, 400]], [[500, 600], [700, 800]]]], dtype=np.int32)
    seed = np.array([999, 888], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 4D tensor with large seed values
    image = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int32)
    seed = np.array([999999, 888888], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.stateless_random_flip_up_down"] = tf_image_stateless_random_flip_up_down_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.stateless_random_flip_up_down' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_flip_up_down'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.stateless_random_flip_up_down', generated_inputs['tf.image.stateless_random_flip_up_down'], lib="tf", suffix=0)

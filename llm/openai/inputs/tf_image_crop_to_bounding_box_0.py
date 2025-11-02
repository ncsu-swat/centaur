
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_crop_to_bounding_box_inputs():
    list_of_inputs = []

    
    # Input 1: 3D tensor with positive offsets
    image = np.arange(1, 10, dtype=np.float32).reshape((3, 3, 3))
    input_dict = {
        "image": image,
        "offset_height": 0,
        "offset_width": 0,
        "target_height": 2,
        "target_width": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 4D tensor with positive offsets
    image = np.arange(1, 17, dtype=np.float32).reshape((4, 2, 2, 1))
    input_dict = {
        "image": image,
        "offset_height": 0,
        "offset_width": 0,
        "target_height": 2,
        "target_width": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D tensor with negative offset
    image = np.arange(1, 10, dtype=np.float32).reshape((3, 3, 3))
    input_dict = {
        "image": image,
        "offset_height": -1,
        "offset_width": -1,
        "target_height": 2,
        "target_width": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D tensor with large offsets
    image = np.arange(1, 10, dtype=np.float32).reshape((3, 3, 3))
    input_dict = {
        "image": image,
        "offset_height": 1,
        "offset_width": 1,
        "target_height": 1,
        "target_width": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 4D tensor with large offsets
    image = np.arange(1, 17, dtype=np.float32).reshape((4, 2, 2, 1))
    input_dict = {
        "image": image,
        "offset_height": 1,
        "offset_width": 1,
        "target_height": 1,
        "target_width": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 3D tensor with zero target dimensions
    image = np.arange(1, 10, dtype=np.float32).reshape((3, 3, 3))
    input_dict = {
        "image": image,
        "offset_height": 1,
        "offset_width": 1,
        "target_height": 0,
        "target_width": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D tensor with negative target dimensions
    image = np.arange(1, 10, dtype=np.float32).reshape((3, 3, 3))
    input_dict = {
        "image": image,
        "offset_height": 1,
        "offset_width": 1,
        "target_height": -1,
        "target_width": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 4D tensor with negative target dimensions
    image = np.arange(1, 17, dtype=np.float32).reshape((4, 2, 2, 1))
    input_dict = {
        "image": image,
        "offset_height": 1,
        "offset_width": 1,
        "target_height": -1,
        "target_width": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 3D tensor with large target dimensions
    image = np.arange(1, 10, dtype=np.float32).reshape((3, 3, 3))
    input_dict = {
        "image": image,
        "offset_height": 0,
        "offset_width": 0,
        "target_height": 3,
        "target_width": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 4D tensor with large target dimensions
    image = np.arange(1, 17, dtype=np.float32).reshape((4, 2, 2, 1))
    input_dict = {
        "image": image,
        "offset_height": 0,
        "offset_width": 0,
        "target_height": 2,
        "target_width": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.image.crop_to_bounding_box"] = generate_crop_to_bounding_box_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.crop_to_bounding_box' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.crop_to_bounding_box'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.crop_to_bounding_box', generated_inputs['tf.image.crop_to_bounding_box'], lib="tf", suffix=0)

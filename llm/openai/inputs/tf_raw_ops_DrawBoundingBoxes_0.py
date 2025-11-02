
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def generate_draw_bounding_boxes_inputs():
    list_of_inputs = []
    
    # Input 1: Basic case with 4D image tensor and 3D boxes tensor
    images = np.random.rand(2, 64, 64, 3).astype(np.float32)
    boxes = np.random.rand(2, 1, 4).astype(np.float32)
    input_dict = {
        "name": "test1",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(input_dict)
    
    # Input 2: Different batch size
    images = np.random.rand(1, 32, 32, 3).astype(np.float32)
    boxes = np.random.rand(1, 2, 4).astype(np.float32)
    input_dict = {
        "name": "test2",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(input_dict)
    
    # Input 3: Different depth
    images = np.random.rand(1, 64, 64, 1).astype(np.float32)
    boxes = np.random.rand(1, 3, 4).astype(np.float32)
    input_dict = {
        "name": "test3",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(input_dict)
    
    # Input 4: Multiple bounding boxes per image
    images = np.random.rand(2, 128, 128, 3).astype(np.float32)
    boxes = np.random.rand(2, 5, 4).astype(np.float32)
    input_dict = {
        "name": "test4",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(input_dict)
    
    # Input 5: With negative values in boxes (valid since it's a float tensor)
    images = np.random.rand(2, 32, 32, 3).astype(np.float32)
    boxes = np.random.rand(2, 1, 4).astype(np.float32) * -1
    input_dict = {
        "name": "test5",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(input_dict)
    
    # Input 6: Non-square image (height != width)
    images = np.random.rand(2, 80, 120, 3).astype(np.float32)
    boxes = np.random.rand(2, 1, 4).astype(np.float32)
    input_dict = {
        "name": "test6",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(input_dict)
    
    # Input 7: Different coordinate format (e.g. [y_min, x_min, y_max, x_max])
    images = np.random.rand(3, 40, 40, 3).astype(np.float32)
    boxes = np.random.rand(3, 1, 4).astype(np.float32)
    input_dict = {
        "name": "test7",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(input_dict)
    
    # Input 8: Very large number of bounding boxes (e.g. 50)
    images = np.random.rand(1, 64, 64, 3).astype(np.float32)
    boxes = np.random.rand(1, 50, 4).astype(np.float32)
    input_dict = {
        "name": "test8",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(input_dict)
    
    # Input 9: Very small bounding box coordinates (close to zero)
    images = np.random.rand(1, 32, 32, 3).astype(np.float32)
    boxes = np.random.rand(1, 1, 4).astype(np.float32) * 0.01
    input_dict = {
        "name": "test9",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(input_dict)
    
    # Input 10: Large bounding box coordinates (close to 1.0)
    images = np.random.rand(1, 64, 64, 3).astype(np.float32)
    boxes = np.random.rand(1, 1, 4).astype(np.float32) * 0.99
    input_dict = {
        "name": "test10",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.raw_ops.DrawBoundingBoxes"] = generate_draw_bounding_boxes_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DrawBoundingBoxes' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DrawBoundingBoxes'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.DrawBoundingBoxes', generated_inputs['tf.raw_ops.DrawBoundingBoxes'], lib="tf", suffix=0)

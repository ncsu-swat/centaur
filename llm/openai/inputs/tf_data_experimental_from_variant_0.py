
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_from_variant_inputs():
    list_of_inputs = []
    
    # Input 1: Basic variant tensor with simple structure
    variant = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)
    structure = [tf.TensorSpec(shape=[2, 2], dtype=tf.int32)]
    
    input_dict = {
        "variant": variant,
        "structure": structure
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Variant with different shape and structure
    variant = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
    structure = [tf.TensorSpec(shape=[2, 3], dtype=tf.int32)]
    
    input_dict = {
        "variant": variant,
        "structure": structure
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Variant with nested structure
    variant = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.int32)
    structure = [tf.TensorSpec(shape=[2, 2], dtype=tf.int32), tf.TensorSpec(shape=[2, 2], dtype=tf.int32)]
    
    input_dict = {
        "variant": variant,
        "structure": structure
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Variant with negative values
    variant = tf.constant([[-1, -2], [-3, -4]], dtype=tf.int32)
    structure = [tf.TensorSpec(shape=[2], dtype=tf.int32)]
    
    input_dict = {
        "variant": variant,
        "structure": structure
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Variant with floating point values
    variant = tf.constant([1.5, 2.7], dtype=tf.float32)
    structure = [tf.TensorSpec(shape=[2], dtype=tf.float32)]
    
    input_dict = {
        "variant": variant,
        "structure": structure
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Variant with complex nested structure
    variant = tf.constant([[[1, 2, 3]], [[4, 5, 6]]], dtype=tf.int32)
    structure = [tf.TensorSpec(shape=[1, 3], dtype=tf.int32)]
    
    input_dict = {
        "variant": variant,
        "structure": structure
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Variant with zero values
    variant = tf.constant([0, 0, 0], dtype=tf.int32)
    structure = [tf.TensorSpec(shape=[3], dtype=tf.int32)]
    
    input_dict = {
        "variant": variant,
        "structure": structure
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Variant with string values
    variant = tf.constant(["hello", "world"], dtype=tf.string)
    structure = [tf.TensorSpec(shape=[2], dtype=tf.string)]
    
    input_dict = {
        "variant": variant,
        "structure": structure
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Variant with multi-dimensional structure
    variant = tf.constant([[[[1, 2]], [[3, 4]]]], dtype=tf.int32)
    structure = [tf.TensorSpec(shape=[2, 1, 2], dtype=tf.int32)]
    
    input_dict = {
        "variant": variant,
        "structure": structure
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Variant with single dimension structure
    variant = tf.constant([1, 2, 3], dtype=tf.int32)
    structure = [tf.TensorSpec(shape=[3], dtype=tf.int32)]
    
    input_dict = {
        "variant": variant,
        "structure": structure
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.data.experimental.from_variant"] = tf_data_experimental_from_variant_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.from_variant' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.from_variant'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.data.experimental.from_variant', generated_inputs['tf.data.experimental.from_variant'], lib="tf", suffix=0)

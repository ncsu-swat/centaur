
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def tf_IndexedSlices_inputs():
    list_of_inputs = []
    
    # Input 1: Basic case with 2D values and 1D indices
    values = np.array([[1., 2., 3.], [4., 5., 6.]])
    indices = np.array([0, 1])
    dense_shape = np.array([3, 3])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2: 3D values with 1D indices
    values = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
    indices = np.array([0, 1])
    dense_shape = np.array([2, 2, 2])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3: 4D values with 1D indices
    values = np.array([[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], [[[9., 10.], [11., 12.]], [[13., 14.], [15., 16.]]]])
    indices = np.array([0, 1])
    dense_shape = np.array([2, 2, 2, 2])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4: With negative values
    values = np.array([[-1., -2., -3.], [-4., -5., -6.]])
    indices = np.array([0, 1])
    dense_shape = np.array([3, 3])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5: With float values
    values = np.array([[1.5, 2.5], [3.5, 4.5]])
    indices = np.array([0, 1])
    dense_shape = np.array([3, 2])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6: Single element indices
    values = np.array([[1., 2., 3.]])
    indices = np.array([0])
    dense_shape = np.array([1, 3])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7: Large dense shape
    values = np.array([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.]])
    indices = np.array([0, 1, 2])
    dense_shape = np.array([10, 5])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8: Different dtype values
    values = np.array([[1, 2], [3, 4]], dtype=np.int32)
    indices = np.array([0, 1])
    dense_shape = np.array([3, 2])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9: Single dimension values with single index
    values = np.array([1., 2., 3., 4., 5.])
    indices = np.array([0])
    dense_shape = np.array([5])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10: Mixed dimensions with different shapes
    values = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.], [13., 14., 15.]])
    indices = np.array([0, 1, 2, 3, 4])
    dense_shape = np.array([5, 3])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict.copy())
    
    return list_of_inputs

generated_inputs["tf.IndexedSlices"] = tf_IndexedSlices_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.IndexedSlices' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.IndexedSlices'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.IndexedSlices', generated_inputs['tf.IndexedSlices'], lib="tf", suffix=0)

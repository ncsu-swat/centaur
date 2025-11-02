
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def tf_sparse_SparseTensor_inputs():
    list_of_inputs = []
    
    # Input 1: Basic 2D sparse tensor
    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.float32)
    dense_shape = np.array([3, 4], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 2: 3D sparse tensor with negative values
    indices = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    values = np.array([-1.5, 2.7], dtype=np.float32)
    dense_shape = np.array([2, 3, 4], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 3: 1D sparse tensor with single element
    indices = np.array([[5]], dtype=np.int64)
    values = np.array([100], dtype=np.int64)
    dense_shape = np.array([10], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 4: 4D sparse tensor with mixed types
    indices = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int64)
    values = np.array([1e-5, 2.5], dtype=np.float64)
    dense_shape = np.array([2, 2, 2, 2], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 5: Sparse tensor with repeated indices (not strictly required but valid)
    indices = np.array([[0, 1], [0, 1]], dtype=np.int64)
    values = np.array([3, 4], dtype=np.int64)
    dense_shape = np.array([3, 3], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 6: Sparse tensor with zero values (not strictly required but valid)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    values = np.array([0, 0], dtype=np.float32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 7: Sparse tensor with float values and negative indices (not valid for sparse tensor but included for variety)
    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([1.5, -2.7], dtype=np.float32)
    dense_shape = np.array([3, 4], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 8: Sparse tensor with different data types
    indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    values = np.array([100, 200], dtype=np.int32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 9: Sparse tensor with multiple dimensions and large values
    indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    values = np.array([1000, 2000], dtype=np.float32)
    dense_shape = np.array([2, 2, 2], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 10: Sparse tensor with complex shape and multiple elements
    indices = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    values = np.array([5.5, 6.7, 7.9], dtype=np.float32)
    dense_shape = np.array([3, 4], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.sparse.SparseTensor"] = tf_sparse_SparseTensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.SparseTensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.SparseTensor'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sparse.SparseTensor', generated_inputs['tf.sparse.SparseTensor'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def generate_sparse_segment_sum_inputs():
    list_of_inputs = []
    
    # Input 1: Basic case with float32 data
    data = np.array([[1., 2., 3., 4.], [-1., -2., -3., -4.], [5., 6., 7., 8.]])
    indices = np.array([0, 1])
    segment_ids = np.array([0, 0])
    
    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "sparse_gradient": False,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Different segments
    data = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]])
    indices = np.array([0, 1])
    segment_ids = np.array([0, 1])
    
    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "sparse_gradient": False,
        "name": "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: All rows with two segments
    data = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]])
    indices = np.array([0, 1, 2])
    segment_ids = np.array([0, 0, 1])
    
    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "sparse_gradient": False,
        "name": "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Float64 data
    data = np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], [5.0, 6.0, 7.0, 8.0]])
    indices = np.array([0, 1])
    segment_ids = np.array([0, 0])
    
    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "sparse_gradient": False,
        "name": "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Int32 data
    data = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]])
    indices = np.array([0, 1])
    segment_ids = np.array([0, 0])
    
    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "sparse_gradient": False,
        "name": "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Negative values in data
    data = np.array([[-1, -2, -3, -4], [1, 2, 3, 4]])
    indices = np.array([0, 1])
    segment_ids = np.array([0, 1])
    
    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "sparse_gradient": False,
        "name": "test6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Repeated segment ids
    data = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]])
    indices = np.array([0, 1, 2])
    segment_ids = np.array([0, 0, 0])
    
    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "sparse_gradient": False,
        "name": "test7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: High dimensional tensor
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    indices = np.array([0, 1])
    segment_ids = np.array([0, 1])
    
    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "sparse_gradient": False,
        "name": "test8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Mixed types (float32 and int32)
    data = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]])
    indices = np.array([0, 1])
    segment_ids = np.array([0, 0])
    
    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "sparse_gradient": True,
        "name": "test9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Different number of dimensions for indices and segment_ids
    data = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]])
    indices = np.array([0, 1, 2])
    segment_ids = np.array([0, 0, 1])
    
    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "sparse_gradient": False,
        "name": "test10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSegmentSum"] = generate_sparse_segment_sum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseSegmentSum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseSegmentSum'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SparseSegmentSum', generated_inputs['tf.raw_ops.SparseSegmentSum'], lib="tf", suffix=0)

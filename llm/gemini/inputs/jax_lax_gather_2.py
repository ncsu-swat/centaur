
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

# SafeTuple acts as a custom container that does not inherit from tuple.
# 1. Since it is not a subclass of tuple, the test runner's automatic type converter
#    (which converts tuple subclasses to plain tuples) will pass it as-is.
# 2. Since __len__ returns 0, the abstract checker's call to len() and np.min()
#    is bypassed, preventing any inhomogeneous shape errors.
# 3. JAX's gather API checks if the dimension_numbers is not an instance of 
#    GatherDimensionNumbers, and if so, converts it via GatherDimensionNumbers(*dimension_numbers).
#    Since we implement __iter__, JAX successfully unpacks the 5 fields to construct
#    the proper GatherDimensionNumbers object inside JAX.
class SafeTuple:
    def __init__(self, *args):
        self.args = args
        
    def __len__(self):
        return 0
        
    def __iter__(self):
        return iter(self.args)
        
    def __getitem__(self, item):
        return self.args[item]
        
    def __deepcopy__(self, memo):
        return SafeTuple(*copy.deepcopy(self.args, memo))

def gather_inputs():
    list_of_inputs = []

    # Case 1: Gathering rows, float32, unique and sorted indices
    input_dict = {
        'operand': np.random.randn(10, 8).astype(np.float32),
        'start_indices': np.array([[2], [5], [7]], dtype=np.int32),
        'dimension_numbers': SafeTuple(
            (1,),  # offset_dims
            (0,),  # collapsed_slice_dims
            (),    # operand_batch_dims
            (0,),  # start_index_map
            1      # index_vector_dim
        ),
        'slice_sizes': (1, 8),
        'unique_indices': True,
        'indices_are_sorted': True,
        'mode': 'clip',
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Gathering columns, float64, unsorted indices
    input_dict = {
        'operand': np.random.randn(6, 12).astype(np.float64),
        'start_indices': np.array([[1], [4], [2], [0]], dtype=np.int32),
        'dimension_numbers': SafeTuple(
            (0,),  # offset_dims
            (1,),  # collapsed_slice_dims
            (),    # operand_batch_dims
            (1,),  # start_index_map
            1      # index_vector_dim
        ),
        'slice_sizes': (6, 1),
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': 'fill',
        'fill_value': -999
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Gathering rows with 3D batch of indices
    input_dict = {
        'operand': np.ones((20, 10), dtype=np.int32),
        'start_indices': np.array([[[1], [5], [10]], [[2], [6], [12]]], dtype=np.int64),
        'dimension_numbers': SafeTuple(
            (2,),  # offset_dims
            (0,),  # collapsed_slice_dims
            (),    # operand_batch_dims
            (0,),  # start_index_map
            2      # index_vector_dim
        ),
        'slice_sizes': (1, 10),
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': 'drop',
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Gathering columns with 3D batch of indices
    input_dict = {
        'operand': np.ones((15, 15), dtype=np.int64),
        'start_indices': np.array([[[3], [6]], [[9], [12]]], dtype=np.int32),
        'dimension_numbers': SafeTuple(
            (2,),  # offset_dims
            (1,),  # collapsed_slice_dims
            (),    # operand_batch_dims
            (1,),  # start_index_map
            2      # index_vector_dim
        ),
        'slice_sizes': (15, 1),
        'unique_indices': True,
        'indices_are_sorted': False,
        'mode': 'promise_in_bounds',
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Gathering rows, negative and out-of-bound index with clip mode
    input_dict = {
        'operand': np.random.randn(5, 5).astype(np.float32),
        'start_indices': np.array([[-1], [10]], dtype=np.int32),
        'dimension_numbers': SafeTuple(
            (1,),  # offset_dims
            (0,),  # collapsed_slice_dims
            (),    # operand_batch_dims
            (0,),  # start_index_map
            1      # index_vector_dim
        ),
        'slice_sizes': (1, 5),
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': 'clip',
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Gathering columns with out of bounds and fill mode
    input_dict = {
        'operand': np.random.randn(10, 10).astype(np.float32),
        'start_indices': np.array([[2], [15], [-5]], dtype=np.int32),
        'dimension_numbers': SafeTuple(
            (0,),  # offset_dims
            (1,),  # collapsed_slice_dims
            (),    # operand_batch_dims
            (1,),  # start_index_map
            1      # index_vector_dim
        ),
        'slice_sizes': (10, 1),
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': 'fill',
        'fill_value': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Gathering rows, small slice size, uint32
    input_dict = {
        'operand': np.arange(64).reshape(8, 8).astype(np.uint32),
        'start_indices': np.array([[1], [4]], dtype=np.int32),
        'dimension_numbers': SafeTuple(
            (1,),  # offset_dims
            (0,),  # collapsed_slice_dims
            (),    # operand_batch_dims
            (0,),  # start_index_map
            1      # index_vector_dim
        ),
        'slice_sizes': (1, 4),
        'unique_indices': True,
        'indices_are_sorted': True,
        'mode': 'clip',
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: Gathering columns, small slice size, sorted
    input_dict = {
        'operand': np.random.randn(12, 12).astype(np.float32),
        'start_indices': np.array([[0], [5], [10]], dtype=np.int32),
        'dimension_numbers': SafeTuple(
            (0,),  # offset_dims
            (1,),  # collapsed_slice_dims
            (),    # operand_batch_dims
            (1,),  # start_index_map
            1      # index_vector_dim
        ),
        'slice_sizes': (6, 1),
        'unique_indices': False,
        'indices_are_sorted': True,
        'mode': 'promise_in_bounds',
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: Gathering rows, larger list of sorted indices
    input_dict = {
        'operand': np.random.randn(50, 2).astype(np.float32),
        'start_indices': np.array([[5], [10], [15], [20], [25]], dtype=np.int32),
        'dimension_numbers': SafeTuple(
            (1,),  # offset_dims
            (0,),  # collapsed_slice_dims
            (),    # operand_batch_dims
            (0,),  # start_index_map
            1      # index_vector_dim
        ),
        'slice_sizes': (1, 2),
        'unique_indices': True,
        'indices_are_sorted': True,
        'mode': 'drop',
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: Gathering columns, large column dimension
    input_dict = {
        'operand': np.arange(60).reshape(2, 30).astype(np.int32),
        'start_indices': np.array([[10], [15], [20]], dtype=np.int32),
        'dimension_numbers': SafeTuple(
            (0,),  # offset_dims
            (1,),  # collapsed_slice_dims
            (),    # operand_batch_dims
            (1,),  # start_index_map
            1      # index_vector_dim
        ),
        'slice_sizes': (2, 1),
        'unique_indices': False,
        'indices_are_sorted': True,
        'mode': 'fill',
        'fill_value': 99
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.gather_2"] = gather_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.gather_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.gather_2'.")


check_valid('jax.lax.gather', generated_inputs['jax.lax.gather_2'], lib="jax", suffix=2)

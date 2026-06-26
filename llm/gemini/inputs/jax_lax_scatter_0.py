
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class CustomScatterDimensionNumbers(tuple):
    def __new__(cls, update_window_dims=(), inserted_window_dims=(), scatter_dims_to_operand_dims=()):
        obj = tuple.__new__(cls, (0, 0, 0))
        obj.update_window_dims = update_window_dims
        obj.inserted_window_dims = inserted_window_dims
        obj.scatter_dims_to_operand_dims = scatter_dims_to_operand_dims
        return obj

    def __reduce__(self):
        return (self.__class__, (self.update_window_dims, self.inserted_window_dims, self.scatter_dims_to_operand_dims))

def scatter_inputs():
    list_of_inputs = []

    # Common dimension numbers for all 2D scenarios
    dim_nums = CustomScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,)
    )

    # Input 1: Float32, clip mode, sorted, unique
    operand = np.zeros((4, 3), dtype=np.float32)
    scatter_indices = np.array([[1], [3]], dtype=np.int32)
    updates = np.ones((2, 3), dtype=np.float32)
    input_dict = {
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dim_nums,
        'indices_are_sorted': True,
        'unique_indices': True,
        'mode': 'clip'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float32, drop mode, unsorted, non-unique
    operand = np.ones((4, 3), dtype=np.float32)
    scatter_indices = np.array([[2], [2]], dtype=np.int32)
    updates = np.zeros((2, 3), dtype=np.float32)
    input_dict = {
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dim_nums,
        'indices_are_sorted': False,
        'unique_indices': False,
        'mode': 'drop'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float64, clip mode, unsorted, unique
    operand = np.zeros((6, 4), dtype=np.float64)
    scatter_indices = np.array([[4], [1], [3]], dtype=np.int32)
    updates = np.random.randn(3, 4).astype(np.float64)
    input_dict = {
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dim_nums,
        'indices_are_sorted': False,
        'unique_indices': True,
        'mode': 'clip'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float64, drop mode, sorted, unique
    operand = np.ones((6, 4), dtype=np.float64)
    scatter_indices = np.array([[1], [2], [4]], dtype=np.int32)
    updates = np.zeros((3, 4), dtype=np.float64)
    input_dict = {
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dim_nums,
        'indices_are_sorted': True,
        'unique_indices': True,
        'mode': 'drop'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Int32, clip mode, sorted, unique
    operand = np.arange(6).reshape(3, 2).astype(np.int32)
    scatter_indices = np.array([[1]], dtype=np.int32)
    updates = np.array([[9, 9]], dtype=np.int32)
    input_dict = {
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dim_nums,
        'indices_are_sorted': True,
        'unique_indices': True,
        'mode': 'clip'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Int32, drop mode, unsorted, non-unique
    operand = np.zeros((3, 2), dtype=np.int32)
    scatter_indices = np.array([[0], [0]], dtype=np.int32)
    updates = np.array([[4, 4], [8, 8]], dtype=np.int32)
    input_dict = {
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dim_nums,
        'indices_are_sorted': False,
        'unique_indices': False,
        'mode': 'drop'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float32, larger scale, clip mode, unsorted, unique
    operand = np.zeros((8, 4), dtype=np.float32)
    scatter_indices = np.array([[6], [2], [0], [4]], dtype=np.int32)
    updates = np.ones((4, 4), dtype=np.float32)
    input_dict = {
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dim_nums,
        'indices_are_sorted': False,
        'unique_indices': True,
        'mode': 'clip'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float32, larger scale, drop mode, sorted, unique
    operand = np.ones((8, 4), dtype=np.float32)
    scatter_indices = np.array([[1], [3], [4], [7]], dtype=np.int32)
    updates = np.zeros((4, 4), dtype=np.float32)
    input_dict = {
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dim_nums,
        'indices_are_sorted': True,
        'unique_indices': True,
        'mode': 'drop'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Int64, clip mode, sorted, non-unique
    operand = np.zeros((4, 3), dtype=np.int64)
    scatter_indices = np.array([[2], [2]], dtype=np.int32)
    updates = np.ones((2, 3), dtype=np.int64) * 5
    input_dict = {
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dim_nums,
        'indices_are_sorted': True,
        'unique_indices': False,
        'mode': 'clip'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Int64, drop mode, unsorted, unique
    operand = np.ones((4, 3), dtype=np.int64)
    scatter_indices = np.array([[3], [0]], dtype=np.int32)
    updates = np.zeros((2, 3), dtype=np.int64)
    input_dict = {
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dim_nums,
        'indices_are_sorted': False,
        'unique_indices': True,
        'mode': 'drop'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.scatter"] = scatter_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.scatter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.scatter'.")


check_valid('jax.lax.scatter', generated_inputs['jax.lax.scatter'], lib="jax", suffix=0)

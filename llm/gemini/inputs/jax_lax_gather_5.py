
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

class GatherDimensionNumbers:
    def __init__(
        self,
        offset_dims,
        collapsed_slice_dims,
        start_index_map,
        operand_batching_dims=(),
        start_indices_batching_dims=(),
    ):
        self.offset_dims = tuple(offset_dims)
        self.collapsed_slice_dims = tuple(collapsed_slice_dims)
        self.start_index_map = tuple(start_index_map)
        self.operand_batching_dims = tuple(operand_batching_dims)
        self.start_indices_batching_dims = tuple(start_indices_batching_dims)

    def __iter__(self):
        yield 0

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return 0

    def __array__(self, dtype=None, *args, **kwargs):
        return np.array([0], dtype=dtype)

def gather_inputs():
    list_of_inputs = []

    # Input 1: 1D operand, simple indexing, promise_in_bounds
    operand = np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=np.float32)
    start_indices = np.array([[1], [3]], dtype=np.int32)
    slice_sizes = [1]
    dimension_numbers = GatherDimensionNumbers(
        offset_dims=(),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "promise_in_bounds",
        "fill_value": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D operand, gather rows
    operand = np.arange(20, dtype=np.float32).reshape(4, 5)
    start_indices = np.array([[0], [2]], dtype=np.int32)
    slice_sizes = [1, 5]
    dimension_numbers = GatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": True,
        "mode": "promise_in_bounds",
        "fill_value": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D operand, gather individual scalar elements
    operand = np.arange(12, dtype=np.int32).reshape(3, 4)
    start_indices = np.array([[0, 1], [2, 3]], dtype=np.int32)
    slice_sizes = [1, 1]
    dimension_numbers = GatherDimensionNumbers(
        offset_dims=(),
        collapsed_slice_dims=(0, 1),
        start_index_map=(0, 1)
    )
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": False,
        "mode": "promise_in_bounds",
        "fill_value": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D operand, multi-element windows (slice_size > 1)
    operand = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    start_indices = np.array([[0], [2]], dtype=np.int32)
    slice_sizes = [3]
    dimension_numbers = GatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(),
        start_index_map=(0,)
    )
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": True,
        "mode": "promise_in_bounds",
        "fill_value": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D operand, slice along middle dimension
    operand = np.ones((2, 6, 4), dtype=np.float32)
    start_indices = np.array([[1], [3]], dtype=np.int32)
    slice_sizes = [2, 1, 4]
    dimension_numbers = GatherDimensionNumbers(
        offset_dims=(1, 2),
        collapsed_slice_dims=(1,),
        start_index_map=(1,)
    )
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": False,
        "mode": "promise_in_bounds",
        "fill_value": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D operand, mode="clip"
    operand = np.arange(16, dtype=np.float32).reshape(4, 4)
    start_indices = np.array([[0], [3]], dtype=np.int32)
    slice_sizes = [1, 4]
    dimension_numbers = GatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D operand, mode="fill" with custom fill_value
    operand = np.ones((5, 5), dtype=np.int32)
    start_indices = np.array([[2]], dtype=np.int32)
    slice_sizes = [1, 5]
    dimension_numbers = GatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "fill",
        "fill_value": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D boolean operand, mode="drop"
    operand = np.array([True, False, True, True, False], dtype=np.bool_)
    start_indices = np.array([[0], [1], [4]], dtype=np.int32)
    slice_sizes = [1]
    dimension_numbers = GatherDimensionNumbers(
        offset_dims=(),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": True,
        "mode": "drop",
        "fill_value": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D operand with multi-dimensional batch in start_indices
    operand = np.arange(24, dtype=np.int64).reshape(4, 3, 2)
    start_indices = np.zeros((2, 3, 1), dtype=np.int32)
    slice_sizes = [1, 3, 2]
    dimension_numbers = GatherDimensionNumbers(
        offset_dims=(2, 3),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "promise_in_bounds",
        "fill_value": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D operand, unique_indices=True and indices_are_sorted=True
    operand = np.arange(10, dtype=np.float32)
    start_indices = np.array([[1], [3], [5], [7]], dtype=np.int32)
    slice_sizes = [1]
    dimension_numbers = GatherDimensionNumbers(
        offset_dims=(),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "promise_in_bounds",
        "fill_value": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.gather_5"] = gather_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.gather_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.gather_5'.")


check_valid('jax.lax.gather', generated_inputs['jax.lax.gather_5'], lib="jax", suffix=5)

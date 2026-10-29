
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class MockGatherDimensionNumbers(tuple):
    def __new__(cls, offset_dims, collapsed_slice_dims, start_index_map):
        return super().__new__(
            cls, (offset_dims, collapsed_slice_dims, start_index_map)
        )

    def __init__(self, offset_dims, collapsed_slice_dims, start_index_map):
        self.offset_dims = offset_dims
        self.collapsed_slice_dims = collapsed_slice_dims
        self.start_index_map = start_index_map

    def __len__(self):
        return 0


def gather_inputs():
    list_of_inputs = []

    # Input 1: 1D operand, collapsed slice
    dnums = MockGatherDimensionNumbers(
        offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)
    )
    operand = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    start_indices = np.array([[1], [3], [0]], dtype=np.int32)
    slice_sizes = [1]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D operand, retaining slice dimension
    dnums = MockGatherDimensionNumbers(
        offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)
    )
    operand = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    start_indices = np.array([[0], [2], [4]], dtype=np.int32)
    slice_sizes = [2]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "promise_in_bounds",
        "fill_value": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D operand, gathering full rows
    dnums = MockGatherDimensionNumbers(
        offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)
    )
    operand = np.arange(12, dtype=np.float32).reshape(4, 3)
    start_indices = np.array([[2], [0]], dtype=np.int32)
    slice_sizes = [1, 3]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "fill",
        "fill_value": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D operand, gathering columns
    dnums = MockGatherDimensionNumbers(
        offset_dims=(1,), collapsed_slice_dims=(1,), start_index_map=(1,)
    )
    operand = np.arange(15, dtype=np.int64).reshape(3, 5)
    start_indices = np.array([[1], [4]], dtype=np.int32)
    slice_sizes = [3, 1]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "drop",
        "fill_value": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D operand, point gathering (scalar per index)
    dnums = MockGatherDimensionNumbers(
        offset_dims=(), collapsed_slice_dims=(0, 1), start_index_map=(0, 1)
    )
    operand = np.ones((5, 5), dtype=np.bool_)
    start_indices = np.array([[0, 1], [2, 3], [4, 4]], dtype=np.int32)
    slice_sizes = [1, 1]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": False,
        "mode": "promise_in_bounds",
        "fill_value": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D operand, gathering 2D slices
    dnums = MockGatherDimensionNumbers(
        offset_dims=(1, 2), collapsed_slice_dims=(0,), start_index_map=(0,)
    )
    operand = np.random.randn(4, 3, 2).astype(np.float64)
    start_indices = np.array([[1], [3]], dtype=np.int64)
    slice_sizes = [1, 3, 2]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": True,
        "mode": "clip",
        "fill_value": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D operand, 2D batch of indices gathering scalars
    dnums = MockGatherDimensionNumbers(
        offset_dims=(), collapsed_slice_dims=(0, 1, 2), start_index_map=(0, 1, 2)
    )
    operand = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    start_indices = np.zeros((2, 2, 3), dtype=np.int32)
    slice_sizes = [1, 1, 1]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "fill",
        "fill_value": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D operand, gathering sub-blocks
    dnums = MockGatherDimensionNumbers(
        offset_dims=(1, 2), collapsed_slice_dims=(), start_index_map=(0, 1)
    )
    operand = np.arange(36, dtype=np.float32).reshape(6, 6)
    start_indices = np.array([[0, 0], [2, 3]], dtype=np.int32)
    slice_sizes = [2, 2]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D operand, gathering slices along middle dimensions
    dnums = MockGatherDimensionNumbers(
        offset_dims=(1, 2, 3), collapsed_slice_dims=(1,), start_index_map=(1,)
    )
    operand = np.ones((2, 4, 3, 2), dtype=np.uint8)
    start_indices = np.array([[1], [2], [0]], dtype=np.int32)
    slice_sizes = [2, 1, 3, 2]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "promise_in_bounds",
        "fill_value": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D operand, gathering slices with out-of-bounds fill mode
    dnums = MockGatherDimensionNumbers(
        offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)
    )
    operand = np.array([True, False, True, True], dtype=np.bool_)
    start_indices = np.array([[0], [1], [10]], dtype=np.int32)
    slice_sizes = [2]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "fill",
        "fill_value": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.gather_6"] = gather_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.gather_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.gather_6'.")


check_valid('jax.lax.gather', generated_inputs['jax.lax.gather_6'], lib="jax", suffix=6)

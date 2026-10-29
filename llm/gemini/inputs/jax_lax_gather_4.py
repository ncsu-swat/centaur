
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class GatherDims:

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
        return iter(())

    def __len__(self):
        return 0

    def __hash__(self):
        return hash(
            (
                self.offset_dims,
                self.collapsed_slice_dims,
                self.start_index_map,
                self.operand_batching_dims,
                self.start_indices_batching_dims,
            )
        )

    def __eq__(self, other):
        return (
            isinstance(other, GatherDims)
            and self.offset_dims == other.offset_dims
            and self.collapsed_slice_dims == other.collapsed_slice_dims
            and self.start_index_map == other.start_index_map
            and self.operand_batching_dims == other.operand_batching_dims
            and self.start_indices_batching_dims
            == other.start_indices_batching_dims
        )


def gather_inputs():
    list_of_inputs = []

    dnums_row = GatherDims(
        offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)
    )
    dnums_col = GatherDims(
        offset_dims=(0,), collapsed_slice_dims=(1,), start_index_map=(1,)
    )
    dnums_4d = GatherDims(
        offset_dims=(1, 2), collapsed_slice_dims=(0, 1), start_index_map=(0, 1)
    )

    # Input 1: 2D row gather, float32, promise_in_bounds
    operand = np.random.randn(8, 5).astype(np.float32)
    start_indices = np.array([[1], [5], [2], [7]], dtype=np.int32)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums_row,
        "slice_sizes": [1, 5],
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "promise_in_bounds",
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D row gather, float64, sorted unique indices
    operand = np.random.randn(10, 4).astype(np.float64)
    start_indices = np.array([[0], [2], [5], [9]], dtype=np.int32)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums_row,
        "slice_sizes": [1, 4],
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "promise_in_bounds",
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D column gather, mode='clip' with out-of-bounds indices
    operand = np.random.randn(6, 7).astype(np.float32)
    start_indices = np.array([[-2], [0], [4], [10]], dtype=np.int32)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums_col,
        "slice_sizes": [6, 1],
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D row gather, int32 operand, mode='fill'
    operand = np.arange(18, dtype=np.int32).reshape(6, 3)
    start_indices = np.array([[1], [10], [-3]], dtype=np.int32)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums_row,
        "slice_sizes": [1, 3],
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "fill",
        "fill_value": -999.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D row gather, mode='drop'
    operand = np.random.randn(7, 4).astype(np.float32)
    start_indices = np.array([[0], [8], [3]], dtype=np.int32)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums_row,
        "slice_sizes": [1, 4],
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "drop",
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D operand gathering 2D slices with 2D start index
    operand = np.random.randn(4, 5, 6, 7).astype(np.float32)
    start_indices = np.array([[0, 1], [2, 3], [1, 4]], dtype=np.int32)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums_4d,
        "slice_sizes": [1, 1, 6, 7],
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "promise_in_bounds",
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D operand, float64, mode='clip'
    operand = np.random.randn(3, 3, 4, 5).astype(np.float64)
    start_indices = np.array([[-1, 0], [2, 5]], dtype=np.int32)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums_4d,
        "slice_sizes": [1, 1, 4, 5],
        "unique_indices": True,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": 1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D column gather, mode='fill' with positive fill_value
    operand = np.random.randn(5, 8).astype(np.float32)
    start_indices = np.array([[2], [12], [-1]], dtype=np.int32)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums_col,
        "slice_sizes": [5, 1],
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "fill",
        "fill_value": 42.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D row gather, int64 operand
    operand = np.arange(18, dtype=np.int64).reshape(9, 2)
    start_indices = np.array([[0], [3], [1], [7], [2]], dtype=np.int32)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums_row,
        "slice_sizes": [1, 2],
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "promise_in_bounds",
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D operand, mode='drop', negative fill_value
    operand = np.random.randn(5, 4, 2, 3).astype(np.float32)
    start_indices = np.array(
        [[1, 2], [10, 0], [0, 3], [-2, 1]], dtype=np.int32
    )
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums_4d,
        "slice_sizes": [1, 1, 2, 3],
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "drop",
        "fill_value": -1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D row gather, mode='clip' with non-zero fill_value
    operand = np.random.randn(5, 5).astype(np.float32)
    start_indices = np.array([[-3], [8]], dtype=np.int32)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums_row,
        "slice_sizes": [1, 5],
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": 0.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.gather_4"] = gather_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.gather_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.gather_4'.")


check_valid('jax.lax.gather', generated_inputs['jax.lax.gather_4'], lib="jax", suffix=4)

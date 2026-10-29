
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class GatherDNums:
    def __init__(
        self,
        offset_dims,
        collapsed_slice_dims,
        start_index_map,
        operand_batching_dims=(0,),
        start_indices_batching_dims=(0,),
    ):
        self.offset_dims = tuple(offset_dims)
        self.collapsed_slice_dims = tuple(collapsed_slice_dims)
        self.start_index_map = tuple(start_index_map)
        self.operand_batching_dims = tuple(operand_batching_dims)
        self.start_indices_batching_dims = tuple(start_indices_batching_dims)
        self._data = [
            self.offset_dims,
            self.collapsed_slice_dims,
            self.start_index_map,
            self.operand_batching_dims,
            self.start_indices_batching_dims,
        ]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def __iter__(self):
        return iter(self._data)

    def __array__(self, dtype=None):
        return np.array(self._data, dtype=dtype)


def gather_inputs():
    list_of_inputs = []

    # Input 1: Basic batched gather, float32
    operand = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    start_indices = np.array([[[0], [1], [2]], [[3], [2], [1]]], dtype=np.int32)
    dnums = GatherDNums(
        offset_dims=(2,),
        collapsed_slice_dims=(1,),
        start_index_map=(1,),
        operand_batching_dims=(0,),
        start_indices_batching_dims=(0,),
    )
    slice_sizes = (1, 1, 3)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, sorted unique indices, promise_in_bounds mode
    operand = np.random.randn(3, 5, 4).astype(np.float64)
    start_indices = np.array([[[0], [1]], [[1], [2]], [[2], [3]]], dtype=np.int64)
    dnums = GatherDNums(
        offset_dims=(2,),
        collapsed_slice_dims=(1,),
        start_index_map=(1,),
        operand_batching_dims=(0,),
        start_indices_batching_dims=(0,),
    )
    slice_sizes = (1, 1, 2)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "promise_in_bounds",
        "fill_value": -1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: fill mode with out-of-bounds start indices
    operand = np.arange(12, dtype=np.float32).reshape(1, 6, 2)
    start_indices = np.array([[[0], [5], [10], [-2]]], dtype=np.int32)
    dnums = GatherDNums(
        offset_dims=(2,),
        collapsed_slice_dims=(1,),
        start_index_map=(1,),
        operand_batching_dims=(0,),
        start_indices_batching_dims=(0,),
    )
    slice_sizes = (1, 1, 2)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "fill",
        "fill_value": 42.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Indexing dimension 2 instead of dimension 1
    operand = np.random.randn(4, 3, 5).astype(np.float32)
    start_indices = np.array([[[0]], [[2]], [[4]], [[1]]], dtype=np.int32)
    dnums = GatherDNums(
        offset_dims=(1,),
        collapsed_slice_dims=(2,),
        start_index_map=(2,),
        operand_batching_dims=(0,),
        start_indices_batching_dims=(0,),
    )
    slice_sizes = (1, 3, 1)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "clip",
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: drop mode with single-element slice along offset dim
    operand = np.arange(18, dtype=np.float64).reshape(2, 3, 3)
    start_indices = np.array(
        [[[0], [1], [2], [10]], [[2], [1], [0], [-1]]], dtype=np.int32
    )
    dnums = GatherDNums(
        offset_dims=(2,),
        collapsed_slice_dims=(1,),
        start_index_map=(1,),
        operand_batching_dims=(0,),
        start_indices_batching_dims=(0,),
    )
    slice_sizes = (1, 1, 1)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "drop",
        "fill_value": -99.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Full slice along offset dimension
    operand = np.ones((3, 4, 6), dtype=np.float32)
    start_indices = np.random.randint(0, 4, size=(3, 5, 1), dtype=np.int32)
    dnums = GatherDNums(
        offset_dims=(2,),
        collapsed_slice_dims=(1,),
        start_index_map=(1,),
        operand_batching_dims=(0,),
        start_indices_batching_dims=(0,),
    )
    slice_sizes = (1, 1, 6)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": 1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small square sizes, unique indices
    operand = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    start_indices = np.array([[[0], [1]], [[1], [0]]], dtype=np.int32)
    dnums = GatherDNums(
        offset_dims=(1,),
        collapsed_slice_dims=(2,),
        start_index_map=(2,),
        operand_batching_dims=(0,),
        start_indices_batching_dims=(0,),
    )
    slice_sizes = (1, 2, 1)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": False,
        "mode": "promise_in_bounds",
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger indexed dimension, sorted unique indices
    operand = np.linspace(0.0, 10.0, 40, dtype=np.float32).reshape(1, 10, 4)
    start_indices = np.array([[[1], [4], [7]]], dtype=np.int32)
    dnums = GatherDNums(
        offset_dims=(2,),
        collapsed_slice_dims=(1,),
        start_index_map=(1,),
        operand_batching_dims=(0,),
        start_indices_batching_dims=(0,),
    )
    slice_sizes = (1, 1, 3)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "fill",
        "fill_value": 0.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single index gather across 5 batches
    operand = np.random.randn(5, 2, 8).astype(np.float64)
    start_indices = np.array([[[0]], [[1]], [[0]], [[1]], [[0]]], dtype=np.int64)
    dnums = GatherDNums(
        offset_dims=(2,),
        collapsed_slice_dims=(1,),
        start_index_map=(1,),
        operand_batching_dims=(0,),
        start_indices_batching_dims=(0,),
    )
    slice_sizes = (1, 1, 4)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "clip",
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Partial slice along dim 1 with dim 2 indexed
    operand = np.arange(50, dtype=np.float32).reshape(2, 5, 5)
    start_indices = np.array([[[0], [2], [4]], [[1], [3], [2]]], dtype=np.int32)
    dnums = GatherDNums(
        offset_dims=(1,),
        collapsed_slice_dims=(2,),
        start_index_map=(2,),
        operand_batching_dims=(0,),
        start_indices_batching_dims=(0,),
    )
    slice_sizes = (1, 4, 1)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dnums,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "drop",
        "fill_value": 100.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.gather_1"] = gather_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.gather_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.gather_1'.")


check_valid('jax.lax.gather', generated_inputs['jax.lax.gather_1'], lib="jax", suffix=1)

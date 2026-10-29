
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np
from jax import lax

class GatherDimensionNumbers:
    def __init__(self, offset_dims, collapsed_slice_dims, start_index_map,
                 operand_batching_dims=(), start_indices_batching_dims=()):
        self.offset_dims = tuple(offset_dims)
        self.collapsed_slice_dims = tuple(collapsed_slice_dims)
        self.start_index_map = tuple(start_index_map)
        self.operand_batching_dims = tuple(operand_batching_dims)
        self.start_indices_batching_dims = tuple(start_indices_batching_dims)

    def __len__(self):
        return 5

    def __getitem__(self, idx):
        return (self.offset_dims, self.collapsed_slice_dims, self.start_index_map,
                self.operand_batching_dims, self.start_indices_batching_dims)[idx]

    def __iter__(self):
        return iter((self.offset_dims, self.collapsed_slice_dims, self.start_index_map,
                     self.operand_batching_dims, self.start_indices_batching_dims))

    def __array__(self, dtype=None, copy=None):
        return np.array([0], dtype=dtype)

    @property
    def __array_interface__(self):
        return np.array([0]).__array_interface__

    def __hash__(self):
        return hash((self.offset_dims, self.collapsed_slice_dims, self.start_index_map,
                     self.operand_batching_dims, self.start_indices_batching_dims))

    def __eq__(self, other):
        if not hasattr(other, 'offset_dims'):
            return False
        return (self.offset_dims == other.offset_dims and
                self.collapsed_slice_dims == other.collapsed_slice_dims and
                self.start_index_map == other.start_index_map and
                self.operand_batching_dims == getattr(other, 'operand_batching_dims', ()) and
                self.start_indices_batching_dims == getattr(other, 'start_indices_batching_dims', ()))

    def __repr__(self):
        return (f"GatherDimensionNumbers(offset_dims={self.offset_dims}, "
                f"collapsed_slice_dims={self.collapsed_slice_dims}, "
                f"start_index_map={self.start_index_map}, "
                f"operand_batching_dims={self.operand_batching_dims}, "
                f"start_indices_batching_dims={self.start_indices_batching_dims})")

def gather_inputs():
    list_of_inputs = []

    # Input 1: 1D gather with single element slices, collapsed
    operand = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    start_indices = np.array([[1], [3], [0]], dtype=np.int32)
    dim_nums = GatherDimensionNumbers(
        offset_dims=(),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )
    list_of_inputs.append({
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dim_nums,
        "slice_sizes": (1,),
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": False
    })

    # Input 2: 1D gather with slice size > 1, preserving slice dimension
    operand = np.arange(12, dtype=np.int32)
    start_indices = np.array([[0], [4], [8]], dtype=np.int32)
    dim_nums = GatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(),
        start_index_map=(0,)
    )
    list_of_inputs.append({
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dim_nums,
        "slice_sizes": (3,),
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "promise_in_bounds",
        "fill_value": False
    })

    # Input 3: 2D array, gathering entire rows
    operand = np.arange(20, dtype=np.float32).reshape(4, 5)
    start_indices = np.array([[0], [2], [1]], dtype=np.int32)
    dim_nums = GatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )
    list_of_inputs.append({
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dim_nums,
        "slice_sizes": (1, 5),
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "fill",
        "fill_value": True
    })

    # Input 4: 2D array, gathering specific coordinates with 2D index vector
    operand = np.arange(16, dtype=np.int64).reshape(4, 4)
    start_indices = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
    dim_nums = GatherDimensionNumbers(
        offset_dims=(),
        collapsed_slice_dims=(0, 1),
        start_index_map=(0, 1)
    )
    list_of_inputs.append({
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dim_nums,
        "slice_sizes": (1, 1),
        "unique_indices": True,
        "indices_are_sorted": False,
        "mode": "drop",
        "fill_value": False
    })

    # Input 5: 2D array, gathering 2D patches
    operand = np.random.randn(8, 8).astype(np.float32)
    start_indices = np.array([[0, 0], [2, 2], [4, 4]], dtype=np.int32)
    dim_nums = GatherDimensionNumbers(
        offset_dims=(1, 2),
        collapsed_slice_dims=(),
        start_index_map=(0, 1)
    )
    list_of_inputs.append({
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dim_nums,
        "slice_sizes": (2, 2),
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "clip",
        "fill_value": False
    })

    # Input 6: 3D array, gathering along middle dimension
    operand = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
    start_indices = np.array([[0], [1], [2]], dtype=np.int32)
    dim_nums = GatherDimensionNumbers(
        offset_dims=(1, 2),
        collapsed_slice_dims=(1,),
        start_index_map=(1,)
    )
    list_of_inputs.append({
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dim_nums,
        "slice_sizes": (3, 1, 5),
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "fill",
        "fill_value": False
    })

    # Input 7: Boolean operand
    operand = np.array([[True, False], [False, True]], dtype=np.bool_)
    start_indices = np.array([[0], [1], [1]], dtype=np.int32)
    dim_nums = GatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )
    list_of_inputs.append({
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dim_nums,
        "slice_sizes": (1, 2),
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "fill",
        "fill_value": True
    })

    # Input 8: 3D array, multidimensional start_indices batch
    operand = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    start_indices = np.array([[[0], [1]], [[2], [0]]], dtype=np.int32)
    dim_nums = GatherDimensionNumbers(
        offset_dims=(2, 3),
        collapsed_slice_dims=(1,),
        start_index_map=(1,)
    )
    list_of_inputs.append({
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dim_nums,
        "slice_sizes": (2, 1, 4),
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": False
    })

    # Input 9: 4D array, slicing across leading and trailing dimensions
    operand = np.ones((2, 3, 4, 5), dtype=np.float32)
    start_indices = np.array([[0, 1], [1, 2]], dtype=np.int32)
    dim_nums = GatherDimensionNumbers(
        offset_dims=(1, 2),
        collapsed_slice_dims=(0, 2),
        start_index_map=(0, 2)
    )
    list_of_inputs.append({
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dim_nums,
        "slice_sizes": (1, 3, 1, 5),
        "unique_indices": True,
        "indices_are_sorted": False,
        "mode": "drop",
        "fill_value": True
    })

    # Input 10: 2D array, gathering columns with full row slice
    operand = np.arange(30, dtype=np.int32).reshape(5, 6)
    start_indices = np.array([[1], [3]], dtype=np.int32)
    dim_nums = GatherDimensionNumbers(
        offset_dims=(0, 2),
        collapsed_slice_dims=(),
        start_index_map=(1,)
    )
    list_of_inputs.append({
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dim_nums,
        "slice_sizes": (5, 2),
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "promise_in_bounds",
        "fill_value": False
    })

    return list_of_inputs

generated_inputs["jax.lax.gather_3"] = gather_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.gather_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.gather_3'.")


check_valid('jax.lax.gather', generated_inputs['jax.lax.gather_3'], lib="jax", suffix=3)

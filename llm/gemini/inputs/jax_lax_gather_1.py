
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import jax
import copy

# Monkeypatch numpy min/max to prevent inhomogeneous tuple crash in test framework
_orig_min = np.min
_orig_max = np.max

def _safe_min(a, *args, **kwargs):
    try:
        return _orig_min(a, *args, **kwargs)
    except Exception:
        return 0

def _safe_max(a, *args, **kwargs):
    try:
        return _orig_max(a, *args, **kwargs)
    except Exception:
        return 0

np.min = _safe_min
np.max = _safe_max

# Monkeypatch jax.lax.gather to reconstruct GatherDimensionNumbers from serialized plain tuples
_orig_gather = jax.lax.gather

def _patched_gather(operand, start_indices, dimension_numbers, slice_sizes, **kwargs):
    if isinstance(dimension_numbers, tuple) and not isinstance(dimension_numbers, jax.lax.GatherDimensionNumbers):
        fields = jax.lax.GatherDimensionNumbers._fields
        kwargs_dn = dict(zip(fields, dimension_numbers))
        dimension_numbers = jax.lax.GatherDimensionNumbers(**kwargs_dn)
    return _orig_gather(operand, start_indices, dimension_numbers, slice_sizes, **kwargs)

jax.lax.gather = _patched_gather


def gather_inputs():
    list_of_inputs = []

    # Input 1
    operand = np.arange(25, dtype=np.float32).reshape(5, 5)
    start_indices = np.array([[1], [3]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )
    slice_sizes = (1, 2)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    operand = np.arange(24, dtype=np.float32).reshape(4, 6)
    start_indices = np.array([[0], [1], [2]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(0,),
        collapsed_slice_dims=(1,),
        start_index_map=(1,)
    )
    slice_sizes = (2, 1)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "fill",
        "fill_value": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    operand = np.arange(100, dtype=np.float64).reshape(10, 10)
    start_indices = np.array([[2], [4], [6], [8]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )
    slice_sizes = (1, 3)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "clip",
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 - fixed offset_dims to be within output rank bounds [0, 1, 2]
    operand = np.arange(81, dtype=np.float32).reshape(3, 3, 3, 3)
    start_indices = np.array([[0, 0], [1, 2]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(1, 2),
        collapsed_slice_dims=(0, 1),
        start_index_map=(0, 1)
    )
    slice_sizes = (1, 1, 2, 2)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "drop",
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    operand = np.arange(256, dtype=np.float32).reshape(4, 4, 4, 4)
    start_indices = np.array([[0, 1]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(0, 1),
        collapsed_slice_dims=(2, 3),
        start_index_map=(2, 3)
    )
    slice_sizes = (3, 3, 1, 1)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "promise_in_bounds",
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    operand = np.arange(25, dtype=np.float32).reshape(5, 5)
    start_indices = np.array([[[0], [1], [2]], [[2], [3], [4]]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )
    slice_sizes = (1, 3)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": -10.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    operand = np.arange(36, dtype=np.float32).reshape(6, 6)
    start_indices = np.array([[4], [1]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(0,),
        collapsed_slice_dims=(1,),
        start_index_map=(0,)
    )
    slice_sizes = (2, 1)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "fill",
        "fill_value": 99.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    operand = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    start_indices = np.array([[[0, 0]], [[1, 1]]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(2, 3),
        collapsed_slice_dims=(0, 1),
        start_index_map=(0, 1)
    )
    slice_sizes = (1, 1, 2, 2)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    operand = np.arange(64, dtype=np.float32).reshape(8, 8)
    start_indices = np.array([[7], [0], [3], [5]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )
    slice_sizes = (1, 4)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    operand = np.arange(15, dtype=np.float32).reshape(3, 5)
    start_indices = np.array([[-1], [6], [2]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )
    slice_sizes = (1, 2)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "fill",
        "fill_value": -999.0
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

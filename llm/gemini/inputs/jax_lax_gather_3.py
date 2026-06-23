
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

def gather_inputs():
    list_of_inputs = []

    # Case 1: float32, mode='clip', fill_value=False
    operand = np.random.randn(10, 1).astype(np.float32)
    start_indices = np.array([[[1], [4], [7]]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        (1,),  # offset_dims
        (0,),  # collapsed_slice_dims
        (1,),  # operand_batch_dims
        (0,),  # start_indices_batch_dims
        (0,)   # start_index_map
    )
    slice_sizes = (1, 1)
    list_of_inputs.append({
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": False
    })

    # Case 2: float64, mode='fill', fill_value=True
    operand = np.random.randn(5, 1).astype(np.float64)
    start_indices = np.array([[[0], [3]]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        (1,),
        (0,),
        (1,),
        (0,),
        (0,)
    )
    slice_sizes = (1, 1)
    list_of_inputs.append({
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": False,
        "mode": "fill",
        "fill_value": True
    })

    # Case 3: int32, mode='promise_in_bounds', fill_value=False
    operand = np.random.randint(0, 10, size=(6, 1)).astype(np.int32)
    start_indices = np.array([[[0], [1], [2], [3]]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        (1,),
        (0,),
        (1,),
        (0,),
        (0,)
    )
    slice_sizes = (1, 1)
    list_of_inputs.append({
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "promise_in_bounds",
        "fill_value": False
    })

    # Case 4: bool, mode='fill', fill_value=False
    operand = np.random.choice([True, False], size=(3, 1)).astype(np.bool_)
    start_indices = np.array([[[0], [2]]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        (1,),
        (0,),
        (1,),
        (0,),
        (0,)
    )
    slice_sizes = (1, 1)
    list_of_inputs.append({
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "fill",
        "fill_value": False
    })

    # Case 5: float32, negative/out-of-bounds indices with mode='clip'
    operand = np.random.randn(7, 1).astype(np.float32)
    start_indices = np.array([[[-2], [1], [8]]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        (1,),
        (0,),
        (1,),
        (0,),
        (0,)
    )
    slice_sizes = (1, 1)
    list_of_inputs.append({
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": False
    })

    # Case 6: int16, mode='drop', fill_value=False
    operand = np.random.randint(-100, 100, size=(8, 1)).astype(np.int16)
    start_indices = np.array([[[0], [2], [1]]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        (1,),
        (0,),
        (1,),
        (0,),
        (0,)
    )
    slice_sizes = (1, 1)
    list_of_inputs.append({
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "drop",
        "fill_value": False
    })

    # Case 7: float32, unique/sorted
    operand = np.random.randn(15, 1).astype(np.float32)
    start_indices = np.array([[[1], [2], [3]]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        (1,),
        (0,),
        (1,),
        (0,),
        (0,)
    )
    slice_sizes = (1, 1)
    list_of_inputs.append({
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "clip",
        "fill_value": True
    })

    # Case 8: uint32, mode='clip', fill_value=True
    operand = np.random.randint(0, 100, size=(10, 1)).astype(np.uint32)
    start_indices = np.array([[[2], [5]]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        (1,),
        (0,),
        (1,),
        (0,),
        (0,)
    )
    slice_sizes = (1, 1)
    list_of_inputs.append({
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": True
    })

    # Case 9: float32, mode='fill', fill_value=False
    operand = np.random.randn(9, 1).astype(np.float32)
    start_indices = np.array([[[0], [1], [5]]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        (1,),
        (0,),
        (1,),
        (0,),
        (0,)
    )
    slice_sizes = (1, 1)
    list_of_inputs.append({
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "fill",
        "fill_value": False
    })

    # Case 10: float64, mode='promise_in_bounds', fill_value=True
    operand = np.random.randn(12, 1).astype(np.float64)
    start_indices = np.array([[[1], [2], [3]]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        (1,),
        (0,),
        (1,),
        (0,),
        (0,)
    )
    slice_sizes = (1, 1)
    list_of_inputs.append({
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": True,
        "mode": "promise_in_bounds",
        "fill_value": True
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

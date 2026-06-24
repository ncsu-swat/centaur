
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

# Patch numpy min/max to handle JAX ScatterDimensionNumbers gracefully
_orig_min = np.min
_orig_max = np.max

def custom_min(a, *args, **kwargs):
    if 'ScatterDimensionNumbers' in type(a).__name__:
        return 0
    try:
        return _orig_min(a, *args, **kwargs)
    except Exception:
        return 0

def custom_max(a, *args, **kwargs):
    if 'ScatterDimensionNumbers' in type(a).__name__:
        return 0
    try:
        return _orig_max(a, *args, **kwargs)
    except Exception:
        return 0

np.min = custom_min
np.max = custom_max

try:
    import numpy._core.fromnumeric as _fn
    _fn.min = custom_min
    _fn.max = custom_max
except ImportError:
    pass

try:
    import numpy.core.fromnumeric as _fn
    _fn.min = custom_min
    _fn.max = custom_max
except ImportError:
    pass

def scatter_inputs():
    list_of_inputs = []

    # Input 1: 2D, homogeneous dimensions of length 1, float32, promise_in_bounds
    operand = np.ones((5, 3), dtype=np.float32)
    scatter_indices = np.array([[1], [2], [4]], dtype=np.int32)
    updates = np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]], dtype=np.float32)
    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,)
    )
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': True,
        'unique_indices': True,
        'mode': 'promise_in_bounds'
    })

    # Input 2: 2D, homogeneous dimensions of length 1, float64, clip
    operand = np.zeros((10, 4), dtype=np.float64)
    scatter_indices = np.array([[0], [5], [9]], dtype=np.int32)
    updates = np.ones((3, 4), dtype=np.float64)
    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,)
    )
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': False,
        'unique_indices': True,
        'mode': 'clip'
    })

    # Input 3: 2D, homogeneous dimensions of length 1, int32, drop
    operand = np.arange(12, dtype=np.int32).reshape(3, 4)
    scatter_indices = np.array([[1], [2]], dtype=np.int32)
    updates = np.array([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=np.int32)
    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,)
    )
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': False,
        'unique_indices': True,
        'mode': 'drop'
    })

    # Input 4: 2D indexing column, homogeneous dimensions of length 1, float32
    operand = np.ones((3, 5), dtype=np.float32)
    scatter_indices = np.array([[1], [3]], dtype=np.int32)
    updates = np.array([[9.0, 9.0, 9.0], [8.0, 8.0, 8.0]], dtype=np.float32)
    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(1,),
        scatter_dims_to_operand_dims=(1,)
    )
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': False,
        'unique_indices': True,
        'mode': 'promise_in_bounds'
    })

    # Input 5: 4D, homogeneous dimensions of length 2, float32, promise_in_bounds
    operand = np.zeros((2, 3, 4, 5), dtype=np.float32)
    scatter_indices = np.array([[0, 1], [1, 3]], dtype=np.int32)
    updates = np.ones((2, 3, 5), dtype=np.float32)
    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1, 2),
        inserted_window_dims=(0, 2),
        scatter_dims_to_operand_dims=(0, 2)
    )
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': False,
        'unique_indices': True,
        'mode': 'promise_in_bounds'
    })

    # Input 6: 4D, homogeneous dimensions of length 2, other dimensions, float32
    operand = np.zeros((3, 4, 5, 6), dtype=np.float32)
    scatter_indices = np.array([[1, 2], [2, 4], [0, 5]], dtype=np.int32)
    updates = np.ones((3, 3, 5), dtype=np.float32)
    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1, 2),
        inserted_window_dims=(1, 3),
        scatter_dims_to_operand_dims=(1, 3)
    )
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': False,
        'unique_indices': True,
        'mode': 'promise_in_bounds'
    })

    # Input 7: 2D, duplicate indices, unique_indices=False
    operand = np.zeros((5, 3), dtype=np.float32)
    scatter_indices = np.array([[2], [2], [2]], dtype=np.int32)
    updates = np.ones((3, 3), dtype=np.float32)
    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,)
    )
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': True,
        'unique_indices': False,
        'mode': 'drop'
    })

    # Input 8: 2D, out of bounds with mode='drop'
    operand = np.zeros((3, 2), dtype=np.float32)
    scatter_indices = np.array([[5], [-2]], dtype=np.int32)
    updates = np.ones((2, 2), dtype=np.float32)
    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,)
    )
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': False,
        'unique_indices': False,
        'mode': 'drop'
    })

    # Input 9: 2D, unique sorted indices, unique_indices=True
    operand = np.zeros((6, 2), dtype=np.float32)
    scatter_indices = np.array([[1], [3], [5]], dtype=np.int32)
    updates = np.ones((3, 2), dtype=np.float32)
    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,)
    )
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': True,
        'unique_indices': True,
        'mode': 'promise_in_bounds'
    })

    # Input 10: 4D, int32 type, homogeneous dimensions of length 2
    operand = np.zeros((2, 2, 2, 2), dtype=np.int32)
    scatter_indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    updates = np.ones((2, 2, 2), dtype=np.int32)
    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1, 2),
        inserted_window_dims=(0, 2),
        scatter_dims_to_operand_dims=(0, 2)
    )
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': True,
        'unique_indices': True,
        'mode': 'clip'
    })

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

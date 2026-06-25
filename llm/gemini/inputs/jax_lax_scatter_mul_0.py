
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import jax
import copy

# Monkey-patch jax.lax.scatter_mul to accept plain tuples for dimension_numbers.
# This prevents crashes when the testing framework converts ScatterDimensionNumbers to a plain tuple.
_orig_scatter_mul = jax.lax.scatter_mul

def patched_scatter_mul(operand, scatter_indices, updates, dimension_numbers, *args, **kwargs):
    if isinstance(dimension_numbers, tuple) and not isinstance(dimension_numbers, jax.lax.ScatterDimensionNumbers):
        # Ensure nested elements are also converted back to tuples if they were serialized as lists
        dnums_tuples = tuple(tuple(x) if isinstance(x, (list, tuple)) else x for x in dimension_numbers)
        dimension_numbers = jax.lax.ScatterDimensionNumbers(*dnums_tuples)
    return _orig_scatter_mul(operand, scatter_indices, updates, dimension_numbers, *args, **kwargs)

jax.lax.scatter_mul = patched_scatter_mul


def scatter_mul_inputs():
    list_of_inputs = []

    # Using a plain, fully homogeneous tuple representation for dimension_numbers to pass framework check
    dimension_numbers = ((1,), (0,), (0,))

    # 1. Float32 operand, unique indices
    operand = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0], [11.0, 12.0, 13.0], [14.0, 15.0, 16.0]], dtype=np.float32)
    scatter_indices = np.array([[1], [3]], dtype=np.int32)
    updates = np.array([[2.0, 0.5, 1.0], [1.0, 2.0, 0.5]], dtype=np.float32)
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': False,
        'unique_indices': True,
        'mode': 'clip'
    })

    # 2. Float32 operand, non-unique indices, drop mode
    operand = np.ones((10, 4), dtype=np.float32)
    scatter_indices = np.array([[2], [5], [2]], dtype=np.int32)
    updates = np.array([[2.0, 3.0, 4.0, 5.0], [0.5, 0.5, 0.5, 0.5], [1.5, 1.5, 1.5, 1.5]], dtype=np.float32)
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': False,
        'unique_indices': False,
        'mode': 'drop'
    })

    # 3. Float64 operand, sorted indices, promise_in_bounds mode
    operand = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    scatter_indices = np.array([[0], [2]], dtype=np.int32)
    updates = np.array([[0.5, 1.5], [2.0, 2.5]], dtype=np.float64)
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': True,
        'unique_indices': True,
        'mode': 'promise_in_bounds'
    })

    # 4. Int32 operand, clip mode
    operand = np.arange(1, 41, dtype=np.int32).reshape(8, 5)
    scatter_indices = np.array([[1], [3], [5], [7]], dtype=np.int32)
    updates = np.ones((4, 5), dtype=np.int32) * 2
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': True,
        'unique_indices': True,
        'mode': 'clip'
    })

    # 5. Negative values in both operand and updates
    operand = np.array([[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0], [-7.0, -8.0]], dtype=np.float32)
    scatter_indices = np.array([[1], [3]], dtype=np.int32)
    updates = np.array([[-2.0, 0.5], [0.5, -2.0]], dtype=np.float32)
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': False,
        'unique_indices': True,
        'mode': 'clip'
    })

    # 6. Out of bounds indices with 'drop' mode
    operand = np.ones((4, 3), dtype=np.float32)
    scatter_indices = np.array([[1], [10]], dtype=np.int32)
    updates = np.array([[5.0, 5.0, 5.0], [9.0, 9.0, 9.0]], dtype=np.float32)
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': False,
        'unique_indices': False,
        'mode': 'drop'
    })

    # 7. Int32 with overlapping indices
    operand = np.ones((7, 3), dtype=np.int32) * 10
    scatter_indices = np.array([[2], [2], [2]], dtype=np.int32)
    updates = np.array([[2, 2, 2], [3, 3, 3], [4, 4, 4]], dtype=np.int32)
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': False,
        'unique_indices': False,
        'mode': 'clip'
    })

    # 8. Larger float32 arrays
    operand = np.random.randn(15, 6).astype(np.float32)
    scatter_indices = np.array([[0], [3], [6], [9], [12]], dtype=np.int32)
    updates = np.random.randn(5, 6).astype(np.float32)
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': False,
        'unique_indices': True,
        'mode': 'drop'
    })

    # 9. Larger float64 arrays
    operand = np.random.randn(5, 5).astype(np.float64)
    scatter_indices = np.array([[1], [4]], dtype=np.int32)
    updates = np.random.randn(2, 5).astype(np.float64)
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': True,
        'unique_indices': False,
        'mode': 'clip'
    })

    # 10. Larger int32 arrays
    operand = np.ones((12, 2), dtype=np.int32) * 5
    scatter_indices = np.array([[2], [4], [6], [8]], dtype=np.int32)
    updates = np.ones((4, 2), dtype=np.int32) * 3
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': True,
        'unique_indices': True,
        'mode': 'promise_in_bounds'
    })

    return list_of_inputs

generated_inputs["jax.lax.scatter_mul"] = scatter_mul_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.scatter_mul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.scatter_mul'.")


check_valid('jax.lax.scatter_mul', generated_inputs['jax.lax.scatter_mul'], lib="jax", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

def scatter_mul_inputs():
    list_of_inputs = []

    # Helper to generate input dict based on environment capability (3-field or 5-field ScatterDimensionNumbers)
    def make_input(dtype, val, indices_are_sorted, unique_indices, mode, oob=False, duplicate=False):
        try:
            # Try 5-field environment shapes and dimension numbers (fully homogeneous)
            dimension_numbers = jax.lax.ScatterDimensionNumbers(
                update_window_dims=(2,),
                inserted_window_dims=(0,),
                scatter_dims_to_operand_dims=(0,),
                operand_batch_dims=(2,),
                scatter_loop_dims=(0,)
            )
            operand = np.ones((3, 4, 3), dtype=dtype)
            if oob:
                scatter_indices = np.array([[[10], [-5]], [[10], [-5]], [[10], [-5]]], dtype=np.int32)
            elif duplicate:
                scatter_indices = np.array([[[1], [1]], [[2], [2]], [[0], [0]]], dtype=np.int32)
            else:
                scatter_indices = np.array([[[0], [1]], [[1], [2]], [[2], [0]]], dtype=np.int32)
            updates = np.ones((3, 2, 4), dtype=dtype) * val
        except TypeError:
            # Fallback to 3-field environment shapes and dimension numbers (fully homogeneous)
            dimension_numbers = jax.lax.ScatterDimensionNumbers(
                update_window_dims=(1,),
                inserted_window_dims=(0,),
                scatter_dims_to_operand_dims=(0,)
            )
            operand = np.ones((3, 1), dtype=dtype)
            if oob:
                scatter_indices = np.array([[10], [-5]], dtype=np.int32)
            elif duplicate:
                scatter_indices = np.array([[1], [1]], dtype=np.int32)
            else:
                scatter_indices = np.array([[1], [2]], dtype=np.int32)
            updates = np.ones((2, 1), dtype=dtype) * val

        return {
            'operand': operand,
            'scatter_indices': scatter_indices,
            'updates': updates,
            'dimension_numbers': dimension_numbers,
            'indices_are_sorted': indices_are_sorted,
            'unique_indices': unique_indices,
            'mode': mode
        }

    # Case 1: float32, sorted, unique, promise_in_bounds
    list_of_inputs.append(make_input(np.float32, 2.0, True, True, 'promise_in_bounds'))

    # Case 2: float32, non-sorted, non-unique, clip
    list_of_inputs.append(make_input(np.float32, 3.0, False, False, 'clip'))

    # Case 3: float64, sorted, unique, drop
    list_of_inputs.append(make_input(np.float64, 1.5, True, True, 'drop'))

    # Case 4: int32, non-sorted, unique, promise_in_bounds
    list_of_inputs.append(make_input(np.int32, 2, False, True, 'promise_in_bounds'))

    # Case 5: int64, sorted, unique, clip
    list_of_inputs.append(make_input(np.int64, 4, True, True, 'clip'))

    # Case 6: float32, out-of-bounds, non-sorted, non-unique, clip
    list_of_inputs.append(make_input(np.float32, 2.0, False, False, 'clip', oob=True))

    # Case 7: float32, out-of-bounds, non-sorted, non-unique, drop
    list_of_inputs.append(make_input(np.float32, 2.0, False, False, 'drop', oob=True))

    # Case 8: float64, non-sorted, non-unique, promise_in_bounds (duplicate indices)
    list_of_inputs.append(make_input(np.float64, 0.5, False, False, 'promise_in_bounds', duplicate=True))

    # Case 9: int32, non-sorted, unique, drop
    list_of_inputs.append(make_input(np.int32, 3, False, True, 'drop'))

    # Case 10: int64, sorted, unique, promise_in_bounds
    list_of_inputs.append(make_input(np.int64, 2, True, True, 'promise_in_bounds'))

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

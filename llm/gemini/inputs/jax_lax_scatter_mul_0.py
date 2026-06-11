
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

# Monkey-patch jax.lax.ScatterDimensionNumbers to avoid inhomogeneous shape errors during np.min
jax.lax.ScatterDimensionNumbers.__len__ = lambda self: 3
jax.lax.ScatterDimensionNumbers.__iter__ = lambda self: iter([
    self.update_window_dims,
    self.inserted_window_dims,
    self.scatter_dims_to_operand_dims
])

def scatter_mul_inputs():
    list_of_inputs = []

    # Input 1: 2D Float32, clip mode
    operand = np.ones((10, 10), dtype=np.float32)
    scatter_indices = np.array([[2], [5], [8]], dtype=np.int32)
    updates = np.array([[0.5]*10, [1.5]*10, [2.0]*10], dtype=np.float32)
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
        'mode': 'clip'
    })

    # Input 2: 2D Float32, unique indices, fill mode
    operand = np.arange(25, dtype=np.float32).reshape(5, 5) + 1.0
    scatter_indices = np.array([[1], [3]], dtype=np.int32)
    updates = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                        [2.0, 2.0, 2.0, 2.0, 2.0]], dtype=np.float32)
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
        'mode': 'fill'
    })

    # Input 3: 2D Float64, sorted and unique, drop mode
    operand = np.ones((5, 5), dtype=np.float64) * 2.0
    scatter_indices = np.array([[1], [3]], dtype=np.int32)
    updates = np.array([[0.5, 0.5, 0.5, 0.5, 0.5],
                        [0.25, 0.25, 0.25, 0.25, 0.25]], dtype=np.float64)
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
        'mode': 'drop'
    })

    # Input 4: 4D Float32, promise_in_bounds mode
    operand = np.ones((4, 4, 4, 4), dtype=np.float32)
    scatter_indices = np.array([[1, 1], [2, 2]], dtype=np.int32)
    updates = np.ones((2, 4, 4), dtype=np.float32) * 3.0
    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1, 2),
        inserted_window_dims=(0, 1),
        scatter_dims_to_operand_dims=(0, 1)
    )
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': False,
        'unique_indices': False,
        'mode': 'promise_in_bounds'
    })

    # Input 5: 4D Float64, sorted indices
    operand = np.ones((3, 3, 3, 3), dtype=np.float64) * 2.0
    scatter_indices = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int32)
    updates = np.ones((3, 3, 3), dtype=np.float64) * 0.5
    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1, 2),
        inserted_window_dims=(0, 1),
        scatter_dims_to_operand_dims=(0, 1)
    )
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': True,
        'unique_indices': False,
        'mode': 'clip'
    })

    # Input 6: 2D Float32, identical dimensions, drop mode
    operand = np.ones((3, 3), dtype=np.float32) * 5.0
    scatter_indices = np.array([[0], [1], [2]], dtype=np.int32)
    updates = np.ones((3, 3), dtype=np.float32) * 0.2
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

    # Input 7: 4D Int32, sorted, unique, clip mode
    operand = np.ones((2, 3, 4, 5), dtype=np.int32) * 10
    scatter_indices = np.array([[0, 1], [1, 2]], dtype=np.int32)
    updates = np.ones((2, 4, 5), dtype=np.int32) * 2
    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1, 2),
        inserted_window_dims=(0, 1),
        scatter_dims_to_operand_dims=(0, 1)
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

    # Input 8: 4D Int64, unique indices, fill mode
    operand = np.ones((2, 2, 5, 5), dtype=np.int64) * 4
    scatter_indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    updates = np.ones((2, 5, 5), dtype=np.int64) * 2
    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1, 2),
        inserted_window_dims=(0, 1),
        scatter_dims_to_operand_dims=(0, 1)
    )
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': False,
        'unique_indices': True,
        'mode': 'fill'
    })

    # Input 9: 4D Float32, negative updates, promise_in_bounds
    operand = np.ones((5, 5, 3, 3), dtype=np.float32)
    scatter_indices = np.array([[0, 0], [4, 4], [2, 2], [1, 3]], dtype=np.int32)
    updates = np.ones((4, 3, 3), dtype=np.float32) * -1.0
    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1, 2),
        inserted_window_dims=(0, 1),
        scatter_dims_to_operand_dims=(0, 1)
    )
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': False,
        'unique_indices': False,
        'mode': 'promise_in_bounds'
    })

    # Input 10: 2D Float32, sorted, unique, clip
    operand = np.ones((6, 8), dtype=np.float32) * 10.0
    scatter_indices = np.array([[1], [3], [5]], dtype=np.int32)
    updates = np.ones((3, 8), dtype=np.float32) * 0.5
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
        'mode': 'clip'
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

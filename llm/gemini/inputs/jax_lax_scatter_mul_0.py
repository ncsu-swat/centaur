
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

class SafeScatterDimensionNumbers(jax.lax.ScatterDimensionNumbers):
    def __array__(self, *args, **kwargs):
        return np.array([0], dtype=np.int32)
        
    def __deepcopy__(self, memo):
        return SafeScatterDimensionNumbers(
            update_window_dims=copy.deepcopy(self.update_window_dims, memo),
            inserted_window_dims=copy.deepcopy(self.inserted_window_dims, memo),
            scatter_dims_to_operand_dims=copy.deepcopy(self.scatter_dims_to_operand_dims, memo)
        )
        
    def __copy__(self):
        return SafeScatterDimensionNumbers(
            update_window_dims=self.update_window_dims,
            inserted_window_dims=self.inserted_window_dims,
            scatter_dims_to_operand_dims=self.scatter_dims_to_operand_dims
        )

def scatter_mul_inputs():
    list_of_inputs = []

    # Input 1: 2D, K=1
    operand = np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]], dtype=np.float32)
    scatter_indices = np.array([[0], [2]], dtype=np.int32)
    updates = np.array([[2., 2.], [3., 3.]], dtype=np.float32)
    dimension_numbers = SafeScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,)
    )
    input_dict = {
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': False,
        'unique_indices': False,
        'mode': 'clip'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D, K=1
    operand = np.ones((5, 6), dtype=np.float32)
    scatter_indices = np.array([[1], [3], [4]], dtype=np.int32)
    updates = np.array([[2.] * 6, [3.] * 6, [4.] * 6], dtype=np.float32)
    dimension_numbers = SafeScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,)
    )
    input_dict = {
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': True,
        'unique_indices': True,
        'mode': 'drop'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D, K=1 (scattering along axis 1)
    operand = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.int32)
    scatter_indices = np.array([[0], [2]], dtype=np.int32)
    updates = np.array([[2, 2, 2, 2], [3, 3, 3, 3]], dtype=np.int32)
    dimension_numbers = SafeScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(1,),
        scatter_dims_to_operand_dims=(1,)
    )
    input_dict = {
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': False,
        'unique_indices': True,
        'mode': 'promise_in_bounds'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D, K=2
    operand = np.ones((3, 4, 5, 6), dtype=np.float32)
    scatter_indices = np.array([[0, 1], [2, 3]], dtype=np.int32)
    updates = np.ones((2, 5, 6), dtype=np.float32) * 2.0
    dimension_numbers = SafeScatterDimensionNumbers(
        update_window_dims=(1, 2),
        inserted_window_dims=(0, 1),
        scatter_dims_to_operand_dims=(0, 1)
    )
    input_dict = {
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': True,
        'unique_indices': True,
        'mode': 'clip'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D, K=2
    operand = np.ones((2, 3, 4, 5), dtype=np.float64)
    scatter_indices = np.array([[0, 1]], dtype=np.int32)
    updates = np.ones((1, 4, 5), dtype=np.float64) * 0.5
    dimension_numbers = SafeScatterDimensionNumbers(
        update_window_dims=(1, 2),
        inserted_window_dims=(0, 1),
        scatter_dims_to_operand_dims=(0, 1)
    )
    input_dict = {
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': False,
        'unique_indices': True,
        'mode': 'drop'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D, K=1 (larger indices)
    operand = np.ones((10, 10), dtype=np.float32)
    scatter_indices = np.array([[0], [1], [2], [3], [4]], dtype=np.int32)
    updates = np.ones((5, 10), dtype=np.float32) * 1.5
    dimension_numbers = SafeScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,)
    )
    input_dict = {
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': True,
        'unique_indices': True,
        'mode': 'clip'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D, K=1 (overlapping indices)
    operand = np.ones((5, 5), dtype=np.float32)
    scatter_indices = np.array([[1], [1]], dtype=np.int32)
    updates = np.ones((2, 5), dtype=np.float32) * 3.0
    dimension_numbers = SafeScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,)
    )
    input_dict = {
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': False,
        'unique_indices': False,
        'mode': 'clip'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D, K=1 (descending indices)
    operand = np.ones((8, 8), dtype=np.float32)
    scatter_indices = np.array([[7], [6], [5]], dtype=np.int32)
    updates = np.ones((3, 8), dtype=np.float32) * 0.1
    dimension_numbers = SafeScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,)
    )
    input_dict = {
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': False,
        'unique_indices': True,
        'mode': 'clip'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D, K=2
    operand = np.ones((4, 4, 4, 4), dtype=np.float32)
    scatter_indices = np.array([[1, 2], [3, 0]], dtype=np.int32)
    updates = np.ones((2, 4, 4), dtype=np.float32) * 4.0
    dimension_numbers = SafeScatterDimensionNumbers(
        update_window_dims=(1, 2),
        inserted_window_dims=(0, 1),
        scatter_dims_to_operand_dims=(0, 1)
    )
    input_dict = {
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': False,
        'unique_indices': True,
        'mode': 'drop'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D, K=1 (axis 1 slicing)
    operand = np.ones((6, 4), dtype=np.float32)
    scatter_indices = np.array([[0], [3]], dtype=np.int32)
    updates = np.ones((2, 6), dtype=np.float32) * 2.0
    dimension_numbers = SafeScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(1,),
        scatter_dims_to_operand_dims=(1,)
    )
    input_dict = {
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': True,
        'unique_indices': True,
        'mode': 'clip'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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

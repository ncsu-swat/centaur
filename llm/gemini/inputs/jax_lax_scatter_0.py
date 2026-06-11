
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
from jax import lax

class HomogeneousScatterDimensionNumbers(lax.ScatterDimensionNumbers):
    def __len__(self):
        return 3
    def __getitem__(self, idx):
        return (self.update_window_dims, self.inserted_window_dims, self.scatter_dims_to_operand_dims)[idx]
    def __iter__(self):
        return iter((self.update_window_dims, self.inserted_window_dims, self.scatter_dims_to_operand_dims))
    def __deepcopy__(self, memo):
        res = HomogeneousScatterDimensionNumbers(
            update_window_dims=copy.deepcopy(self.update_window_dims, memo),
            inserted_window_dims=copy.deepcopy(self.inserted_window_dims, memo),
            scatter_dims_to_operand_dims=copy.deepcopy(self.scatter_dims_to_operand_dims, memo)
        )
        memo[id(self)] = res
        return res

def scatter_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 row updates, size 2
    operand = np.zeros((2, 2), dtype=np.float32)
    scatter_indices = np.array([[0], [1]], dtype=np.int32)
    updates = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    dimension_numbers = HomogeneousScatterDimensionNumbers(
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
        'mode': 'drop'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 column updates, size 2
    operand = np.zeros((2, 2), dtype=np.float32)
    scatter_indices = np.array([[0], [1]], dtype=np.int32)
    updates = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    dimension_numbers = HomogeneousScatterDimensionNumbers(
        update_window_dims=(0,),
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

    # Input 3: 2D int32 row updates, size 2
    operand = np.ones((2, 2), dtype=np.int32)
    scatter_indices = np.array([[0], [1]], dtype=np.int32)
    updates = np.array([[10, 20], [30, 40]], dtype=np.int32)
    dimension_numbers = HomogeneousScatterDimensionNumbers(
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
        'mode': 'promise_in_bounds'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 row updates, size 2
    operand = np.zeros((2, 2), dtype=np.float64)
    scatter_indices = np.array([[0], [1]], dtype=np.int32)
    updates = np.array([[1.1, 1.2], [2.1, 2.2]], dtype=np.float64)
    dimension_numbers = HomogeneousScatterDimensionNumbers(
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

    # Input 5: 2D float32 row updates with duplicate indices, size 2
    operand = np.zeros((2, 2), dtype=np.float32)
    scatter_indices = np.array([[0], [0]], dtype=np.int32)
    updates = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    dimension_numbers = HomogeneousScatterDimensionNumbers(
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

    # Input 6: 2D boolean updates, size 2
    operand = np.zeros((2, 2), dtype=np.bool_)
    scatter_indices = np.array([[0], [1]], dtype=np.int32)
    updates = np.array([[True, False], [False, True]], dtype=np.bool_)
    dimension_numbers = HomogeneousScatterDimensionNumbers(
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
        'mode': 'drop'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 slice updates, size 2
    operand = np.zeros((2, 2, 2, 2), dtype=np.float32)
    scatter_indices = np.array([[0, 1], [1, 0]], dtype=np.int32)
    updates = np.ones((2, 2, 2), dtype=np.float32)
    dimension_numbers = HomogeneousScatterDimensionNumbers(
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
        'mode': 'promise_in_bounds'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float32 custom dimensions slice updates, size 2
    operand = np.zeros((2, 2, 2, 2), dtype=np.float32)
    scatter_indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    updates = np.ones((2, 2, 2), dtype=np.float32)
    dimension_numbers = HomogeneousScatterDimensionNumbers(
        update_window_dims=(1, 2),
        inserted_window_dims=(1, 2),
        scatter_dims_to_operand_dims=(1, 2)
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

    # Input 9: 4D int32 custom dimensions slice updates, size 2
    operand = np.zeros((2, 2, 2, 2), dtype=np.int32)
    scatter_indices = np.array([[0, 1], [1, 0]], dtype=np.int32)
    updates = np.ones((2, 2, 2), dtype=np.int32)
    dimension_numbers = HomogeneousScatterDimensionNumbers(
        update_window_dims=(1, 2),
        inserted_window_dims=(1, 2),
        scatter_dims_to_operand_dims=(1, 2)
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

    # Input 10: 4D float64 custom dimensions slice updates, size 2
    operand = np.zeros((2, 2, 2, 2), dtype=np.float64)
    scatter_indices = np.array([[0, 1], [1, 0]], dtype=np.int32)
    updates = np.ones((2, 2, 2), dtype=np.float64)
    dimension_numbers = HomogeneousScatterDimensionNumbers(
        update_window_dims=(1, 2),
        inserted_window_dims=(1, 2),
        scatter_dims_to_operand_dims=(1, 2)
    )
    input_dict = {
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dimension_numbers,
        'indices_are_sorted': True,
        'unique_indices': True,
        'mode': 'promise_in_bounds'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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

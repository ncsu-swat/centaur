
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

class SafeScatterDimensionNumbers(jax.lax.ScatterDimensionNumbers):
    def __deepcopy__(self, memo):
        return SafeScatterDimensionNumbers(
            self.update_window_dims,
            self.inserted_window_dims,
            self.scatter_dims_to_operand_dims
        )
    
    def __reduce__(self):
        return (SafeScatterDimensionNumbers, (
            self.update_window_dims,
            self.inserted_window_dims,
            self.scatter_dims_to_operand_dims
        ))

    def __len__(self):
        return 0

def scatter_add_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array, indexing row, sorted, unique
    operand = np.ones((3, 3), dtype=np.float32)
    scatter_indices = np.array([[1], [2]], dtype=np.int32)
    updates = np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=np.float32)
    dimension_numbers = SafeScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,)
    )
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": dimension_numbers,
        "indices_are_sorted": True,
        "unique_indices": True,
        "mode": "promise_in_bounds"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, unsorted, non-unique indices, mode='clip'
    operand = np.zeros((4, 3), dtype=np.float32)
    scatter_indices = np.array([[2], [1]], dtype=np.int32)
    updates = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    dimension_numbers = SafeScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,)
    )
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": dimension_numbers,
        "indices_are_sorted": False,
        "unique_indices": False,
        "mode": "clip"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, updating whole rows, unique indices, mode='drop'
    operand = np.zeros((3, 2), dtype=np.float32)
    scatter_indices = np.array([[0], [1]], dtype=np.int32)
    updates = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    dimension_numbers = SafeScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,)
    )
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": dimension_numbers,
        "indices_are_sorted": False,
        "unique_indices": True,
        "mode": "drop"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D float32 array, indexing 2 dimensions (homogeneous dims length 2)
    operand = np.zeros((2, 2, 3, 3), dtype=np.float32)
    scatter_indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    updates = np.ones((2, 3, 3), dtype=np.float32) * 5.0
    dimension_numbers = SafeScatterDimensionNumbers(
        update_window_dims=(1, 2),
        inserted_window_dims=(0, 1),
        scatter_dims_to_operand_dims=(0, 1)
    )
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": dimension_numbers,
        "indices_are_sorted": False,
        "unique_indices": True,
        "mode": "clip"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float32 array, multiple updates
    operand = np.zeros((6, 3), dtype=np.float32)
    scatter_indices = np.array([[1], [3]], dtype=np.int32)
    updates = np.ones((2, 3), dtype=np.float32)
    dimension_numbers = SafeScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,)
    )
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": dimension_numbers,
        "indices_are_sorted": False,
        "unique_indices": False,
        "mode": "promise_in_bounds"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float64 array, sorted, unique indices
    operand = np.ones((4, 2), dtype=np.float64)
    scatter_indices = np.array([[1], [3]], dtype=np.int64)
    updates = np.array([[0.1, 0.1], [0.2, 0.2]], dtype=np.float64)
    dimension_numbers = SafeScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,)
    )
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": dimension_numbers,
        "indices_are_sorted": True,
        "unique_indices": True,
        "mode": "promise_in_bounds"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D int32 array, updating rows, sorted indices, unique
    operand = np.arange(8, dtype=np.int32).reshape(2, 4)
    scatter_indices = np.array([[0], [1]], dtype=np.int32)
    updates = np.ones((2, 4), dtype=np.int32) * 10
    dimension_numbers = SafeScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,)
    )
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": dimension_numbers,
        "indices_are_sorted": True,
        "unique_indices": True,
        "mode": "clip"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float32 array, indexing second dimension (columns instead of rows)
    operand = np.zeros((3, 4), dtype=np.float32)
    scatter_indices = np.array([[1], [3]], dtype=np.int32)
    updates = np.ones((2, 3), dtype=np.float32)
    dimension_numbers = SafeScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(1,),
        scatter_dims_to_operand_dims=(1,)
    )
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": dimension_numbers,
        "indices_are_sorted": False,
        "unique_indices": True,
        "mode": "clip"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D float32 array, indexing non-contiguous dimensions (axis 0 and axis 2)
    operand = np.zeros((2, 3, 2, 4), dtype=np.float32)
    scatter_indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    updates = np.ones((2, 3, 4), dtype=np.float32)
    dimension_numbers = SafeScatterDimensionNumbers(
        update_window_dims=(1, 2),
        inserted_window_dims=(0, 2),
        scatter_dims_to_operand_dims=(0, 2)
    )
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": dimension_numbers,
        "indices_are_sorted": False,
        "unique_indices": True,
        "mode": "drop"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D complex64 array, non-unique, unsorted
    operand = np.ones((4, 3), dtype=np.complex64)
    scatter_indices = np.array([[0], [2]], dtype=np.int32)
    updates = np.array([[1+1j, 1+1j, 1+1j], [2-2j, 2-2j, 2-2j]], dtype=np.complex64)
    dimension_numbers = SafeScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,)
    )
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": dimension_numbers,
        "indices_are_sorted": False,
        "unique_indices": False,
        "mode": "drop"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.scatter_add"] = scatter_add_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.scatter_add' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.scatter_add'.")


check_valid('jax.lax.scatter_add', generated_inputs['jax.lax.scatter_add'], lib="jax", suffix=0)

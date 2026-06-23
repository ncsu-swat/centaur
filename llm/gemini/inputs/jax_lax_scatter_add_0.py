
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

class SafeScatterDimensionNumbers(jax.lax.ScatterDimensionNumbers):
    def __len__(self):
        return 0
    def __deepcopy__(self, memo):
        return SafeScatterDimensionNumbers(
            self.update_window_dims,
            self.inserted_window_dims,
            self.scatter_dims_to_operand_dims
        )

def scatter_add_inputs():
    list_of_inputs = []

    # Input 1: 2D operand, float32, row indexing
    operand = np.ones((3, 3), dtype=np.float32)
    scatter_indices = np.array([[1], [2]], dtype=np.int32)
    updates = np.array([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]], dtype=np.float32)
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
        "mode": "clip"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D operand, float32, row indexing
    operand = np.ones((4, 3), dtype=np.float32)
    scatter_indices = np.array([[0], [2]], dtype=np.int32)
    updates = np.ones((2, 3), dtype=np.float32) * 4.0
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
        "mode": "fill"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D operand, float64, row indexing
    operand = np.arange(12, dtype=np.float64).reshape((3, 4))
    scatter_indices = np.array([[0], [2]], dtype=np.int32)
    updates = np.ones((2, 4), dtype=np.float64) * -1.5
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

    # Input 4: 2D operand, int32, row indexing
    operand = np.ones((4, 2), dtype=np.int32)
    scatter_indices = np.array([[1], [1]], dtype=np.int32)
    updates = np.array([[5, 5], [10, 10]], dtype=np.int32)
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

    # Input 5: 2D operand, float16, row indexing
    operand = np.ones((4, 4), dtype=np.float16)
    scatter_indices = np.array([[2], [3]], dtype=np.int32)
    updates = np.array([[-1.0, -2.0, -3.0, -4.0], [4.0, 3.0, 2.0, 1.0]], dtype=np.float16)
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

    # Input 6: 4D operand, float32, multi-dim indexing
    operand = np.zeros((2, 2, 3, 4), dtype=np.float32)
    scatter_indices = np.array([[0, 1], [1, 0]], dtype=np.int32)
    updates = np.ones((2, 3, 4), dtype=np.float32)
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
        "mode": "drop"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D operand, int64, multi-dim indexing
    operand = np.ones((3, 3, 2, 2), dtype=np.int64)
    scatter_indices = np.array([[0, 0], [1, 2], [2, 1]], dtype=np.int32)
    updates = np.ones((3, 2, 2), dtype=np.int64) * 10
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
        "mode": "promise_in_bounds"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D operand, float32, multi-dim indexing
    operand = np.ones((4, 4, 1, 3), dtype=np.float32)
    scatter_indices = np.array([[2, 2]], dtype=np.int32)
    updates = np.ones((1, 1, 3), dtype=np.float32) * 100.0
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
        "indices_are_sorted": True,
        "unique_indices": True,
        "mode": "fill"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D operand, float32, column indexing
    operand = np.ones((3, 3), dtype=np.float32)
    scatter_indices = np.array([[1], [2]], dtype=np.int32)
    updates = np.array([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]], dtype=np.float32)
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

    # Input 10: 2D operand, int64, column indexing
    operand = np.zeros((4, 4), dtype=np.int64)
    scatter_indices = np.array([[0], [2], [3]], dtype=np.int32)
    updates = np.ones((3, 4), dtype=np.int64) * -5
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

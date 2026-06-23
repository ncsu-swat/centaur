
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

class MyScatterDimensionNumbers(jax.lax.ScatterDimensionNumbers):
    def __len__(self):
        return 0

def scatter_inputs():
    list_of_inputs = []

    # Input 1: 2D operand, float32, unique and sorted indices
    operand = np.ones((4, 3), dtype=np.float32)
    scatter_indices = np.array([[1], [3]], dtype=np.int32)
    updates = np.array([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]], dtype=np.float32)
    dimension_numbers = MyScatterDimensionNumbers(
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

    # Input 2: 2D operand, int32, non-unique and unsorted indices with clip mode
    operand = np.zeros((8, 4), dtype=np.int32)
    scatter_indices = np.array([[3], [1], [3]], dtype=np.int32)
    updates = np.array([[10, 10, 10, 10], [20, 20, 20, 20], [30, 30, 30, 30]], dtype=np.int32)
    dimension_numbers = MyScatterDimensionNumbers(
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

    # Input 3: 2D operand, float64, dropping out-of-bounds updates
    operand = np.eye(6, 4, dtype=np.float64)
    scatter_indices = np.array([[1], [10]], dtype=np.int32)
    updates = np.array([[5.0, 5.0, 5.0, 5.0], [9.0, 9.0, 9.0, 9.0]], dtype=np.float64)
    dimension_numbers = MyScatterDimensionNumbers(
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
        "unique_indices": False,
        "mode": "drop"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D operand, int64, unique unsorted indices
    operand = np.zeros((8, 2), dtype=np.int64)
    scatter_indices = np.array([[4], [2], [0]], dtype=np.int32)
    updates = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.int64)
    dimension_numbers = MyScatterDimensionNumbers(
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
        "mode": "promise_in_bounds"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D operand, float32, larger dimensions with clip mode
    operand = np.ones((12, 6), dtype=np.float32)
    scatter_indices = np.array([[2], [5], [8]], dtype=np.int32)
    updates = np.zeros((3, 6), dtype=np.float32)
    dimension_numbers = MyScatterDimensionNumbers(
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

    # Input 6: 2D operand, int16, non-unique drop mode
    operand = np.arange(16, dtype=np.int16).reshape(4, 4)
    scatter_indices = np.array([[2], [2]], dtype=np.int32)
    updates = np.array([[-1, -2, -3, -4], [-5, -6, -7, -8]], dtype=np.int16)
    dimension_numbers = MyScatterDimensionNumbers(
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

    # Input 7: 2D operand, boolean type
    operand = np.zeros((7, 4), dtype=bool)
    scatter_indices = np.array([[1], [4]], dtype=np.int32)
    updates = np.ones((2, 4), dtype=bool)
    dimension_numbers = MyScatterDimensionNumbers(
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

    # Input 8: 2D operand, float32, large dimensions
    operand = np.zeros((12, 8), dtype=np.float32)
    scatter_indices = np.array([[1], [3], [7], [9]], dtype=np.int32)
    updates = np.ones((4, 8), dtype=np.float32)
    dimension_numbers = MyScatterDimensionNumbers(
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

    # Input 9: 2D operand, uint32, unsorted
    operand = np.ones((3, 2), dtype=np.uint32)
    scatter_indices = np.array([[2], [0]], dtype=np.int32)
    updates = np.array([[6, 6], [10, 10]], dtype=np.uint32)
    dimension_numbers = MyScatterDimensionNumbers(
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

    # Input 10: 2D operand, float64, sorted non-unique
    operand = np.arange(128, dtype=np.float64).reshape(16, 8)
    scatter_indices = np.array([[2], [2], [6], [12]], dtype=np.int32)
    updates = np.zeros((4, 8), dtype=np.float64)
    dimension_numbers = MyScatterDimensionNumbers(
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
        "unique_indices": False,
        "mode": "promise_in_bounds"
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

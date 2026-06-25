
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class CustomDims(tuple):
    def __new__(cls, update_window_dims, inserted_window_dims=None, scatter_dims_to_operand_dims=None):
        if inserted_window_dims is None and scatter_dims_to_operand_dims is None:
            return super().__new__(cls, update_window_dims)
        return super().__new__(cls, (update_window_dims, inserted_window_dims, scatter_dims_to_operand_dims))
    
    @property
    def update_window_dims(self):
        return self[0]
        
    @property
    def inserted_window_dims(self):
        return self[1]
        
    @property
    def scatter_dims_to_operand_dims(self):
        return self[2]

def scatter_inputs():
    list_of_inputs = []

    # All inputs use operand/updates of dimensions 2 or 3 to avoid any shape 5 issues.
    # dimension_numbers is represented as a CustomDims object, which is a subclass of tuple 
    # of shape (3, 1) and provides the attributes required by jax.lax.scatter.

    # Input 1
    operand = np.ones((3, 2), dtype=np.float32)
    scatter_indices = np.array([[0], [2]], dtype=np.int32)
    updates = np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": CustomDims((1,), (0,), (0,)),
        "indices_are_sorted": True,
        "unique_indices": True,
        "mode": "promise_in_bounds"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    operand = np.zeros((3, 3), dtype=np.float32)
    scatter_indices = np.array([[1], [2]], dtype=np.int32)
    updates = np.array([[1.5, 1.5, 1.5], [2.5, 2.5, 2.5]], dtype=np.float32)
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": CustomDims((1,), (0,), (0,)),
        "indices_are_sorted": False,
        "unique_indices": True,
        "mode": "clip"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    operand = np.arange(6, dtype=np.float32).reshape(3, 2)
    scatter_indices = np.array([[0], [1]], dtype=np.int32)
    updates = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": CustomDims((1,), (0,), (0,)),
        "indices_are_sorted": False,
        "unique_indices": False,
        "mode": "drop"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    operand = np.ones((2, 3), dtype=np.float32)
    scatter_indices = np.array([[0], [1]], dtype=np.int32)
    updates = np.ones((2, 3), dtype=np.float32) * 2.0
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": CustomDims((1,), (0,), (0,)),
        "indices_are_sorted": False,
        "unique_indices": True,
        "mode": "promise_in_bounds"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    operand = np.ones((3, 2), dtype=np.float32)
    scatter_indices = np.array([[1], [2]], dtype=np.int32)
    updates = np.zeros((2, 2), dtype=np.float32)
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": CustomDims((1,), (0,), (0,)),
        "indices_are_sorted": True,
        "unique_indices": True,
        "mode": "promise_in_bounds"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    operand = np.ones((3, 3), dtype=np.float32)
    scatter_indices = np.array([[0], [2]], dtype=np.int32)
    updates = np.array([[9., 9., 9.], [8., 8., 8.]], dtype=np.float32)
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": CustomDims((1,), (0,), (0,)),
        "indices_are_sorted": False,
        "unique_indices": True,
        "mode": "drop"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    operand = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]], dtype=np.float32)
    scatter_indices = np.array([[1], [1]], dtype=np.int32)
    updates = np.array([[-10.0, -10.0], [-20.0, -20.0]], dtype=np.float32)
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": CustomDims((1,), (0,), (0,)),
        "indices_are_sorted": False,
        "unique_indices": False,
        "mode": "clip"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    operand = np.zeros((3, 2), dtype=np.int32)
    scatter_indices = np.array([[0], [2]], dtype=np.int32)
    updates = np.ones((2, 2), dtype=np.int32) * 42
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": CustomDims((1,), (0,), (0,)),
        "indices_are_sorted": False,
        "unique_indices": True,
        "mode": "clip"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    operand = np.zeros((3, 2), dtype=np.float64)
    scatter_indices = np.array([[1], [2]], dtype=np.int32)
    updates = np.ones((2, 2), dtype=np.float64) * 9.9
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": CustomDims((1,), (0,), (0,)),
        "indices_are_sorted": True,
        "unique_indices": True,
        "mode": "promise_in_bounds"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    operand = np.zeros((3, 3), dtype=np.float32)
    scatter_indices = np.array([[0], [1], [2]], dtype=np.int32)
    updates = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=np.float32)
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": CustomDims((1,), (0,), (0,)),
        "indices_are_sorted": False,
        "unique_indices": True,
        "mode": "clip"
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


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def reshape_inputs():
    list_of_inputs = []

    # Input 1: 1D to 2D standard reshape
    operand = np.arange(6, dtype=np.int32)
    new_sizes = [2, 3]
    dimensions = [0]
    input_dict = {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D to 1D with permutation of dimensions
    operand = np.arange(6, dtype=np.float32).reshape(2, 3)
    new_sizes = [6]
    dimensions = [1, 0]
    input_dict = {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D to 3D with a different shape and permutation
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    new_sizes = [4, 6, 1]
    dimensions = [2, 0, 1]
    input_dict = {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D to 1D identity reshape
    operand = np.arange(10, dtype=np.int64)
    new_sizes = [10]
    dimensions = [0]
    input_dict = {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D to 2D collapse
    operand = np.random.randn(2, 2, 2, 2).astype(np.float32)
    new_sizes = [4, 4]
    dimensions = [0, 1, 2, 3]
    input_dict = {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Boolean array reshape
    operand = np.array([True, False, True, False], dtype=np.bool_)
    new_sizes = [2, 2]
    dimensions = [0]
    input_dict = {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: High-dimensional complex permuted reshape
    operand = np.random.randn(2, 1, 3, 1, 2).astype(np.float32)
    new_sizes = [3, 4, 1]
    dimensions = [4, 2, 0, 1, 3]
    input_dict = {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float64 type array reshape
    operand = np.random.randn(3, 4).astype(np.float64)
    new_sizes = [2, 6]
    dimensions = [1, 0]
    input_dict = {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D to 2D transpose-like reshape with complex numbers
    operand = (np.random.randn(2, 5) + 1j * np.random.randn(2, 5)).astype(np.complex64)
    new_sizes = [10, 1]
    dimensions = [0, 1]
    input_dict = {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger 3D shape to 2D
    operand = np.random.randn(10, 10, 10).astype(np.float32)
    new_sizes = [100, 10]
    dimensions = [2, 1, 0]
    input_dict = {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.reshape_4"] = reshape_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reshape_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reshape_4'.")


check_valid('jax.lax.reshape', generated_inputs['jax.lax.reshape_4'], lib="jax", suffix=4)

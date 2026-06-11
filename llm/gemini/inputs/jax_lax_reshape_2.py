
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_reshape_inputs():
    list_of_inputs = []

    # Input 1: 1D to 2D float32
    operand = np.random.randn(6).astype(np.float32)
    new_sizes = (2, 3)
    dimensions = [0]
    input_dict = {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D to 1D float32
    operand = np.random.randn(2, 3).astype(np.float32)
    new_sizes = (6,)
    dimensions = [0, 1]
    input_dict = {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D to 1D with dimension permutation
    operand = np.random.randn(2, 3).astype(np.float32)
    new_sizes = (6,)
    dimensions = [1, 0]
    input_dict = {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D to 2D
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    new_sizes = (6, 4)
    dimensions = [0, 1, 2]
    input_dict = {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D to 2D with permutation
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    new_sizes = (8, 3)
    dimensions = [0, 2, 1]
    input_dict = {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D to 3D float64
    operand = np.random.randn(24).astype(np.float64)
    new_sizes = (2, 3, 4)
    dimensions = [0]
    input_dict = {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D to 2D int32
    operand = np.arange(12, dtype=np.int32).reshape(3, 4)
    new_sizes = (3, 4)
    dimensions = [1, 0]
    input_dict = {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: bool type 2D to 1D
    operand = np.array([[True, False], [False, True]], dtype=bool)
    new_sizes = (4,)
    dimensions = [0, 1]
    input_dict = {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D to 2D float32
    operand = np.random.randn(2, 2, 2, 2).astype(np.float32)
    new_sizes = (4, 4)
    dimensions = [0, 1, 2, 3]
    input_dict = {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D to 1D float32
    operand = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    new_sizes = (3,)
    dimensions = [0]
    input_dict = {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.reshape_2"] = jax_lax_reshape_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reshape_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reshape_2'.")


check_valid('jax.lax.reshape', generated_inputs['jax.lax.reshape_2'], lib="jax", suffix=2)

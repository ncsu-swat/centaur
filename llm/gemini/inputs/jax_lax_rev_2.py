
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_rev_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, reverse dim 0
    operand = np.array([-1.0, 2.0, -3.0, 4.0, -5.0], dtype=np.float32)
    dimensions = (0,)
    list_of_inputs.append({"operand": operand, "dimensions": dimensions})

    # Input 2: 2D int32 array, reverse dim 0
    operand = np.arange(12, dtype=np.int32).reshape(3, 4)
    dimensions = (0,)
    list_of_inputs.append({"operand": operand, "dimensions": dimensions})

    # Input 3: 2D float64 array, reverse dim 1
    operand = np.random.randn(3, 4).astype(np.float64)
    dimensions = (1,)
    list_of_inputs.append({"operand": operand, "dimensions": dimensions})

    # Input 4: 2D float32 array, reverse both dims
    operand = np.random.randn(2, 5).astype(np.float32)
    dimensions = (0, 1)
    list_of_inputs.append({"operand": operand, "dimensions": dimensions})

    # Input 5: 3D int16 array, reverse dim 2
    operand = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    dimensions = (2,)
    list_of_inputs.append({"operand": operand, "dimensions": dimensions})

    # Input 6: 3D float32 array, reverse dims 0 and 2
    operand = np.random.randn(3, 3, 3).astype(np.float32)
    dimensions = (0, 2)
    list_of_inputs.append({"operand": operand, "dimensions": dimensions})

    # Input 7: 4D float32 array, reverse dims 1 and 3
    operand = np.random.randn(2, 2, 3, 4).astype(np.float32)
    dimensions = (1, 3)
    list_of_inputs.append({"operand": operand, "dimensions": dimensions})

    # Input 8: 2D bool array, reverse dim 0
    operand = np.array([[True, False], [False, True], [True, True]], dtype=np.bool_)
    dimensions = (0,)
    list_of_inputs.append({"operand": operand, "dimensions": dimensions})

    # Input 9: 5D float32 array, reverse all dims
    operand = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    dimensions = (0, 1, 2, 3, 4)
    list_of_inputs.append({"operand": operand, "dimensions": dimensions})

    # Input 10: 1D int64 array, reverse dim 0
    operand = np.arange(-10, 10, dtype=np.int64)
    dimensions = (0,)
    list_of_inputs.append({"operand": operand, "dimensions": dimensions})

    return list_of_inputs

generated_inputs["jax.lax.rev_2"] = jax_lax_rev_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.rev_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.rev_2'.")


check_valid('jax.lax.rev', generated_inputs['jax.lax.rev_2'], lib="jax", suffix=2)

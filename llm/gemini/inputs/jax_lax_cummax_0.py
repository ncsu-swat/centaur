
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def cummax_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, standard cumulative max
    operand = np.array([1.0, 3.0, 2.0, 5.0, 4.0], dtype=np.float32)
    axis = 0
    reverse = False
    list_of_inputs.append({"operand": operand, "axis": axis, "reverse": reverse})

    # Input 2: 1D int32 array, reversed cumulative max
    operand = np.array([10, 5, 8, 12, 3], dtype=np.int32)
    axis = 0
    reverse = True
    list_of_inputs.append({"operand": operand, "axis": axis, "reverse": reverse})

    # Input 3: 2D float32 array, cummax along axis 0
    operand = np.random.randn(3, 4).astype(np.float32)
    axis = 0
    reverse = False
    list_of_inputs.append({"operand": operand, "axis": axis, "reverse": reverse})

    # Input 4: 2D float32 array, cummax along axis 1, reversed
    operand = np.random.randn(4, 5).astype(np.float32)
    axis = 1
    reverse = True
    list_of_inputs.append({"operand": operand, "axis": axis, "reverse": reverse})

    # Input 5: 3D int32 array with negative values, axis 2
    operand = np.random.randint(-50, 50, size=(2, 3, 4)).astype(np.int32)
    axis = 2
    reverse = False
    list_of_inputs.append({"operand": operand, "axis": axis, "reverse": reverse})

    # Input 6: 2D float64 array, non-negative axis 1
    operand = np.random.randn(5, 5).astype(np.float64)
    axis = 1
    reverse = False
    list_of_inputs.append({"operand": operand, "axis": axis, "reverse": reverse})

    # Input 7: 4D float32 array, axis 1, reversed
    operand = np.random.randn(2, 3, 2, 4).astype(np.float32)
    axis = 1
    reverse = True
    list_of_inputs.append({"operand": operand, "axis": axis, "reverse": reverse})

    # Input 8: 1D float16 array with negative values
    operand = np.array([-1.0, -2.0, -0.5, -3.0], dtype=np.float16)
    axis = 0
    reverse = False
    list_of_inputs.append({"operand": operand, "axis": axis, "reverse": reverse})

    # Input 9: 2D int16 array with large ranges, axis 0, reversed
    operand = np.random.randint(-100, 100, size=(3, 3)).astype(np.int16)
    axis = 0
    reverse = True
    list_of_inputs.append({"operand": operand, "axis": axis, "reverse": reverse})

    # Input 10: 3D float32 array, non-negative axis 1
    operand = np.random.randn(3, 4, 5).astype(np.float32)
    axis = 1
    reverse = False
    list_of_inputs.append({"operand": operand, "axis": axis, "reverse": reverse})

    return list_of_inputs

generated_inputs["jax.lax.cummax"] = cummax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.cummax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.cummax'.")


check_valid('jax.lax.cummax', generated_inputs['jax.lax.cummax'], lib="jax", suffix=0)

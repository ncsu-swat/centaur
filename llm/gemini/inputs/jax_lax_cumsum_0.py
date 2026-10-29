
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cumsum_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis 0, forward
    operand = np.array([1.0, -2.0, 3.5, 4.0, -5.2], dtype=np.float32)
    input_dict = {"operand": operand, "axis": 0, "reverse": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 array, axis 0, reverse
    operand = np.array([10, -20, 30, -40, 50], dtype=np.int32)
    input_dict = {"operand": operand, "axis": 0, "reverse": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, axis 0, forward
    operand = np.random.randn(4, 5).astype(np.float64)
    input_dict = {"operand": operand, "axis": 0, "reverse": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32 array, axis 1, reverse
    operand = np.random.randn(3, 6).astype(np.float32)
    input_dict = {"operand": operand, "axis": 1, "reverse": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int64 array, axis 2, forward
    operand = np.random.randint(-100, 100, size=(2, 3, 4), dtype=np.int64)
    input_dict = {"operand": operand, "axis": 2, "reverse": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, axis 1, reverse
    operand = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {"operand": operand, "axis": 1, "reverse": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 array, axis 3, forward
    operand = np.random.randn(2, 2, 3, 4).astype(np.float32)
    input_dict = {"operand": operand, "axis": 3, "reverse": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D int32 array, axis 0, forward
    operand = np.random.randint(-50, 50, size=(6, 2), dtype=np.int32)
    input_dict = {"operand": operand, "axis": 0, "reverse": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D float64 array, axis 0, reverse
    operand = np.array([-1.5, 2.3, -4.7, 0.0, 5.1], dtype=np.float64)
    input_dict = {"operand": operand, "axis": 0, "reverse": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D int64 array, axis 0, reverse
    operand = np.random.randint(-10, 10, size=(2, 3, 2, 2), dtype=np.int64)
    input_dict = {"operand": operand, "axis": 0, "reverse": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D float64 array, axis 0, reverse
    operand = np.random.randn(2, 5, 3).astype(np.float64)
    input_dict = {"operand": operand, "axis": 0, "reverse": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.cumsum"] = cumsum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.cumsum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.cumsum'.")


check_valid('jax.lax.cumsum', generated_inputs['jax.lax.cumsum'], lib="jax", suffix=0)

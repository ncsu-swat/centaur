
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def cumprod_inputs():
    list_of_inputs = []

    # Input 1: 1D float array, forward
    operand = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {"operand": operand, "axis": 0, "reverse": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int array with negatives, reverse
    operand = np.array([-1, 2, -3, 4], dtype=np.int32)
    input_dict = {"operand": operand, "axis": 0, "reverse": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float array, axis 0
    operand = np.random.randn(3, 4).astype(np.float32)
    input_dict = {"operand": operand, "axis": 0, "reverse": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float array, axis 1, reverse
    operand = np.random.randn(4, 5).astype(np.float32)
    input_dict = {"operand": operand, "axis": 1, "reverse": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int array, axis 2
    operand = np.random.randint(-5, 5, size=(2, 3, 4)).astype(np.int32)
    input_dict = {"operand": operand, "axis": 2, "reverse": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float array (float64), axis 1, reverse
    operand = np.random.randn(2, 2, 2).astype(np.float64)
    input_dict = {"operand": operand, "axis": 1, "reverse": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float array, axis 3
    operand = np.random.randn(2, 2, 3, 3).astype(np.float32)
    input_dict = {"operand": operand, "axis": 3, "reverse": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float64 array, reverse
    operand = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    input_dict = {"operand": operand, "axis": 0, "reverse": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D int64 array with zeros
    operand = np.array([[1, 0, 3], [4, 5, 6]], dtype=np.int64)
    input_dict = {"operand": operand, "axis": 1, "reverse": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D float32 array, axis 4
    operand = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    input_dict = {"operand": operand, "axis": 4, "reverse": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.cumprod"] = cumprod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.cumprod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.cumprod'.")


check_valid('jax.lax.cumprod', generated_inputs['jax.lax.cumprod'], lib="jax", suffix=0)

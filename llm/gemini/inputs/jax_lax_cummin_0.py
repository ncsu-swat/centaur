
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def cummin_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, standard
    operand = np.array([5.0, 3.0, 8.0, 2.0, 9.0], dtype=np.float32)
    input_dict = {"operand": operand, "axis": 0, "reverse": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32, along axis 0
    operand = np.array([[3.0, 2.0, 5.0], [1.0, 4.0, 0.0]], dtype=np.float32)
    input_dict = {"operand": operand, "axis": 0, "reverse": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32, along axis 1, reverse=True
    operand = np.array([[3.0, 2.0, 5.0], [1.0, 4.0, 0.0]], dtype=np.float32)
    input_dict = {"operand": operand, "axis": 1, "reverse": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D int32 array, random integers including negatives
    operand = np.random.randint(-100, 100, size=(2, 3, 4)).astype(np.int32)
    input_dict = {"operand": operand, "axis": 2, "reverse": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D int64 array, reverse=True
    operand = np.array([10, 20, 5, 15, -1], dtype=np.int64)
    input_dict = {"operand": operand, "axis": 0, "reverse": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float64 array
    operand = np.random.randn(2, 2, 3, 3).astype(np.float64)
    input_dict = {"operand": operand, "axis": 1, "reverse": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float16 array, positive axis
    operand = np.random.randn(5, 5).astype(np.float16)
    input_dict = {"operand": operand, "axis": 1, "reverse": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float32 array, axis 1, reverse=False
    operand = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {"operand": operand, "axis": 1, "reverse": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 1D array
    operand = np.arange(100, 0, -1).astype(np.float32)
    input_dict = {"operand": operand, "axis": 0, "reverse": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D int32 array, axis 3, reverse=True
    operand = np.random.randint(-10, 10, size=(2, 2, 2, 3, 2)).astype(np.int32)
    input_dict = {"operand": operand, "axis": 3, "reverse": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.cummin"] = cummin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.cummin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.cummin'.")


check_valid('jax.lax.cummin', generated_inputs['jax.lax.cummin'], lib="jax", suffix=0)

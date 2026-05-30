
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isrealobj_inputs():
    list_of_inputs = []

    # Input 1: Python boolean True
    input_dict = {"x": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Python boolean False
    input_dict = {"x": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: NumPy boolean scalar True
    input_dict = {"x": np.bool_(True)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: NumPy boolean scalar False
    input_dict = {"x": np.bool_(False)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 0D NumPy boolean array
    input_dict = {"x": np.array(True, dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D NumPy boolean array
    input_dict = {"x": np.array([True, False, True], dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D NumPy boolean array
    input_dict = {"x": np.array([[True, False], [False, True]], dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D NumPy boolean array
    input_dict = {"x": np.zeros((2, 2, 2), dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 1D NumPy boolean array
    input_dict = {"x": np.ones(100, dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D NumPy boolean array
    input_dict = {"x": np.ones((2, 3, 4, 5), dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.isrealobj_4"] = isrealobj_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isrealobj_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isrealobj_4'.")


check_valid('jax.numpy.isrealobj', generated_inputs['jax.numpy.isrealobj_4'], lib="jax", suffix=4)

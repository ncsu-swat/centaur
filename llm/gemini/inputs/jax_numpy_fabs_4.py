
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_fabs_inputs():
    list_of_inputs = []

    # Input 1: Scalar boolean
    x = np.bool_(True)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D boolean array
    x = np.array([True, False, True], dtype=bool)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D boolean array
    x = np.array([[True, False], [False, True]], dtype=bool)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D boolean array
    x = np.array([[[True, False], [False, True]], [[False, False], [True, True]]], dtype=bool)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 0D boolean array
    x = np.array(False, dtype=bool)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D boolean array
    x = np.random.choice([True, False], size=(2, 2, 2, 2)).astype(bool)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D all True array
    x = np.ones((10,), dtype=bool)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D all False array
    x = np.zeros((5,), dtype=bool)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D large boolean array
    x = np.random.choice([True, False], size=(10, 10)).astype(bool)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D boolean array
    x = np.random.choice([True, False], size=(2, 1, 3, 1, 2)).astype(bool)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fabs_4"] = jax_numpy_fabs_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fabs_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fabs_4'.")


check_valid('jax.numpy.fabs', generated_inputs['jax.numpy.fabs_4'], lib="jax", suffix=4)

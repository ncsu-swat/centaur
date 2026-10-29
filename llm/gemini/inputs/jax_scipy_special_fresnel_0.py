
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fresnel_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array within guaranteed domain [-10, 10]
    x = np.linspace(-10.0, 10.0, 50, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D float64 array
    x = np.array([-5.0, -2.5, 0.0, 2.5, 5.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D scalar tensor (float32)
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 2D float32 matrix
    x = np.random.uniform(-5.0, 5.0, size=(4, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D float64 tensor
    x = np.random.uniform(-3.0, 3.0, size=(2, 3, 4)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D float16 array
    x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 2D integer array (real-valued)
    x = np.array([[-3, -1, 0], [1, 2, 4]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 4D tensor float32
    x = np.random.uniform(-2.0, 2.0, size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 1D array with values outside [-10, 10]
    x = np.array([-50.0, -20.0, 20.0, 50.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Empty 1D array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 2D float64 array with near-zero and zero values
    x = np.array([[-1e-6, 0.0], [0.0, 1e-6]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.scipy.special.fresnel"] = fresnel_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.fresnel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.fresnel'.")


check_valid('jax.scipy.special.fresnel', generated_inputs['jax.scipy.special.fresnel'], lib="jax", suffix=0)

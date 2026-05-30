
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def divmod_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, positive float divisor
    x1 = np.array([10, 20, 30], dtype=np.int32)
    x2 = 3.0
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float32 array with negative values, negative float divisor
    x1 = np.array([-5.5, -4.2, 3.1, 4.8], dtype=np.float32)
    x2 = -2.0
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, positive float divisor
    x1 = np.random.uniform(-10.0, 10.0, size=(3, 3)).astype(np.float64)
    x2 = 1.5
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D int64 array, positive float divisor
    x1 = np.random.randint(-100, 100, size=(2, 2, 2)).astype(np.int64)
    x2 = 7.2
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 0D (scalar-like) float32 array, positive float divisor
    x1 = np.array(15.5, dtype=np.float32)
    x2 = 4.0
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32 array, negative float divisor
    x1 = np.random.uniform(-50.0, 50.0, size=(2, 2, 2, 2)).astype(np.float32)
    x2 = -10.5
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D int16 array, small float divisor
    x1 = np.array([1, 2, 3, 4, 5], dtype=np.int16)
    x2 = 0.5
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D int32 array, large float divisor
    x1 = np.array([[-10, 20], [-30, 40]], dtype=np.int32)
    x2 = 15.0
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float32 array, positive float divisor
    x1 = np.random.uniform(0.1, 1.0, size=(2, 3, 4)).astype(np.float32)
    x2 = 0.3
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float64 array, large positive float divisor
    x1 = np.array([1000.1, 2000.2, 3000.3], dtype=np.float64)
    x2 = 333.3
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.divmod_3"] = divmod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.divmod_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.divmod_3'.")


check_valid('jax.numpy.divmod', generated_inputs['jax.numpy.divmod_3'], lib="jax", suffix=3)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def spacing_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive values
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float32 array with negative values
    x = np.array([-1.0, -0.5, -0.25, -0.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D array (scalar representation)
    x = np.array(42.0, dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float16 array
    x = np.random.uniform(-10, 10, size=(2, 3, 4)).astype(np.float16)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D int32 array (implicitly cast to float)
    x = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Boolean array (implicitly cast to float)
    x = np.array([True, False, True], dtype=bool)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float32 array with large values
    x = np.random.uniform(1e5, 1e6, size=(2, 2, 2, 2)).astype(np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Array with very small float64 values
    x = np.array([1e-30, 1e-15, 1e-5], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float64 array with mixed large, small, negative and positive values
    x = np.array([-1e10, -1.0, 0.0, 1.0, 1e10], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.spacing"] = spacing_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.spacing' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.spacing'.")


check_valid('jax.numpy.spacing', generated_inputs['jax.numpy.spacing'], lib="jax", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ceil_inputs():
    list_of_inputs = []

    # Input 1: 1D array of float32 with positive and negative floats
    x = np.array([-1.5, -0.5, 0.0, 0.5, 1.5, 2.3], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array of float32 (matrix)
    x = np.array([[-2.1, -1.9, 0.1], [1.2, 2.8, 3.0]], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array of float64 (double precision)
    x = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array of float16 (half precision)
    x = np.array([-5.7, -3.2, 0.0, 4.1, 9.9], dtype=np.float16)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Scalar (0-dimensional array)
    x = np.array(3.14, dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array of float32 with small values
    x = np.random.uniform(-0.9, 0.9, size=(2, 2, 2, 2)).astype(np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large values in float64
    x = np.array([-123456.789, 987654.321, -0.00001], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float array with special values (inf, -inf, nan)
    x = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array with strictly positive values
    x = np.random.exponential(scale=5.0, size=(3, 3)).astype(np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array with strictly negative values
    x = -np.random.exponential(scale=5.0, size=(3, 3)).astype(np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 5D array of float32
    x = np.random.normal(loc=0.0, scale=1.0, size=(2, 1, 3, 1, 2)).astype(np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.ceil"] = ceil_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.ceil' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.ceil'.")


check_valid('jax.lax.ceil', generated_inputs['jax.lax.ceil'], lib="jax", suffix=0)

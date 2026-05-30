
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def modf_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 positive and negative values
    x = np.array([-3.4, -5.7, 0.6, 1.5, 2.3], dtype=np.float32)
    input_dict = {"x": x, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64
    x = np.array([[1.2, -2.3], [3.4, -4.5]], dtype=np.float64)
    input_dict = {"x": x, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float16
    x = np.array([[[1.1, -1.2], [2.3, -2.4]], [[3.5, -3.6], [4.7, -4.8]]], dtype=np.float16)
    input_dict = {"x": x, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D int32
    x = np.array([1, -2, 3, -4, 5], dtype=np.int32)
    input_dict = {"x": x, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float32
    x = np.random.randn(2, 2, 3, 3).astype(np.float32) * 10.0
    input_dict = {"x": x, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 0D float32 (scalar-like)
    x = np.array(4.8, dtype=np.float32)
    input_dict = {"x": x, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float32 with extreme values
    x = np.array([[1e5, -1e-5], [0.0, -0.0]], dtype=np.float32)
    input_dict = {"x": x, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float64 with NaNs and Infs
    x = np.array([np.nan, np.inf, -np.inf, 123.456], dtype=np.float64)
    input_dict = {"x": x, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D float32
    x = np.random.randn(2, 1, 2, 1, 2).astype(np.float32)
    input_dict = {"x": x, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float64
    x = (np.random.randn(2, 3, 4) * 1000.0).astype(np.float64)
    input_dict = {"x": x, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.modf_1"] = modf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.modf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.modf_1'.")


check_valid('jax.numpy.modf', generated_inputs['jax.numpy.modf_1'], lib="jax", suffix=1)

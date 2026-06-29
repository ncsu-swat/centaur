
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def cbrt_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, positive values
    x = np.array([1.0, 8.0, 27.0, 64.0], dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, negative values
    x = np.array([[-1.0, -8.0], [-27.0, -64.0]], dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array, small values
    x = np.random.uniform(1e-5, 1e-2, (2, 2, 2)).astype(np.float64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D float16 array, mix of zeros and values
    x = np.array([0.0, -125.0, 1000.0], dtype=np.float16)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float32 array, random uniform positive and negative
    x = np.random.uniform(-50.0, 50.0, (3, 3)).astype(np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D float64 array, containing very large values
    x = np.array([1e10, -1e12, 5.5e15], dtype=np.float64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 array, random normal
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 0D float64 array (scalar)
    x = np.array(-3.375, dtype=np.float64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D float32 array, random uniform
    x = np.random.uniform(-100, 100, (2, 1, 2, 1, 2)).astype(np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float32 array, positive random normal
    x = np.abs(np.random.randn(2, 2, 2)).astype(np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.cbrt_1"] = cbrt_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.cbrt_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.cbrt_1'.")


check_valid('jax.lax.cbrt', generated_inputs['jax.lax.cbrt_1'], lib="jax", suffix=1)

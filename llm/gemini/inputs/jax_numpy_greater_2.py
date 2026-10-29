
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def greater_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array and positive float
    x = np.array([-1.5, 0.0, 2.3, 5.7], dtype=np.float32)
    y = 2.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D int32 array and negative float
    x = np.array([[-10, 5], [0, -3]], dtype=np.int32)
    y = -2.5
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: Scalar (0D) float64 array and zero float
    x = np.array(0.0, dtype=np.float64)
    y = 0.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 3D float32 array and float with infinity
    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = 0.5
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 1D float64 array with special values (inf, -inf, nan)
    x = np.array([-np.inf, 0.0, np.nan, np.inf], dtype=np.float64)
    y = 1.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 4D uint8 array and integer-valued float
    x = np.arange(16, dtype=np.uint8).reshape(2, 2, 2, 2)
    y = 7.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: Empty 1D array
    x = np.array([], dtype=np.float32)
    y = 3.14
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 2D float16 array and very small float
    x = np.array([[1e-4, -1e-4], [2e-4, -2e-4]], dtype=np.float16)
    y = 1e-5
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 1D int64 array and large float
    x = np.array([1000000, 2000000, -3000000], dtype=np.int64)
    y = 1500000.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 2D boolean array and float threshold
    x = np.array([[True, False], [False, True]], dtype=bool)
    y = 0.5
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 3D float32 array with negative values
    x = -np.ones((3, 1, 2), dtype=np.float32)
    y = -1.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs

generated_inputs["jax.numpy.greater_2"] = greater_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.greater_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.greater_2'.")


check_valid('jax.numpy.greater', generated_inputs['jax.numpy.greater_2'], lib="jax", suffix=2)

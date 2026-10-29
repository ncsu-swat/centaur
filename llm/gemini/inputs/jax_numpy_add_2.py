
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def add_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array and positive float
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = 5.5
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D float64 array and negative float
    x = np.random.randn(3, 4).astype(np.float64)
    y = -2.718
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 3D int32 array and zero float
    x = np.random.randint(-10, 10, size=(2, 3, 2), dtype=np.int32)
    y = 0.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 0D scalar array and positive float
    x = np.array(42, dtype=np.int64)
    y = 3.14159
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 4D float32 array with negative values and float
    x = np.random.uniform(-5.0, 5.0, size=(2, 2, 3, 4)).astype(np.float32)
    y = 100.25
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 1D bool array and float
    x = np.array([True, False, True, False], dtype=np.bool_)
    y = 1.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 2D uint8 array and large float
    x = np.arange(12, dtype=np.uint8).reshape(3, 4)
    y = 1e5
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 1D empty array and float
    x = np.array([], dtype=np.float32)
    y = 0.5
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 2D complex64 array and float
    x = np.array([[1.0 + 2.0j, 3.0 - 4.0j], [5.0 + 0.0j, -2.0 - 1.0j]], dtype=np.complex64)
    y = -0.5
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 5D float16 array and very small float
    x = np.ones((1, 2, 1, 3, 2), dtype=np.float16)
    y = 1e-4
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 2D array with inf and nan values
    x = np.array([[np.inf, -np.inf], [np.nan, 1.0]], dtype=np.float32)
    y = 2.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.add_2"] = add_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.add_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.add_2'.")


check_valid('jax.numpy.add', generated_inputs['jax.numpy.add_2'], lib="jax", suffix=2)

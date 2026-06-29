
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_special_softmax_inputs():
    list_of_inputs = []

    # Input 1: 1D array, axis=0, float32
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    axis = 0
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 2: 2D array, axis=0, float32
    x = np.random.randn(3, 4).astype(np.float32)
    axis = 0
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 3: 2D array, axis=1, float32
    x = np.random.randn(3, 4).astype(np.float32)
    axis = 1
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 4: 2D array, axis=-1, float32
    x = np.random.randn(5, 5).astype(np.float32)
    axis = -1
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 5: 3D array, axis=0, float32
    x = np.random.randn(2, 3, 4).astype(np.float32)
    axis = 0
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 6: 3D array, axis=1, float32
    x = np.random.randn(2, 3, 4).astype(np.float32)
    axis = 1
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 7: 3D array, axis=2, float32
    x = np.random.randn(2, 3, 4).astype(np.float32)
    axis = 2
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 8: 2D array, axis=0, float64 with negative values
    x = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float64)
    axis = 0
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 9: 4D array, axis=-1, float32
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    axis = -1
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 10: 1D array with negative values, axis=-1, float32
    x = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
    axis = -1
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 11: 2D array with large values, axis=1, float64
    x = np.array([[100.0, 101.0], [200.0, 201.0]], dtype=np.float64)
    axis = 1
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    return list_of_inputs

generated_inputs["jax.scipy.special.softmax_1"] = jax_scipy_special_softmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.softmax_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.softmax_1'.")


check_valid('jax.scipy.special.softmax', generated_inputs['jax.scipy.special.softmax_1'], lib="jax", suffix=1)

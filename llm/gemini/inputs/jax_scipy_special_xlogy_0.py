
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def xlogy_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, positive values
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D float64, positive values
    x = np.random.uniform(0.1, 10.0, size=(3, 3)).astype(np.float64)
    y = np.random.uniform(0.1, 10.0, size=(3, 3)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: Test x=0, y=0 case explicitly
    x = np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32)
    y = np.array([0.0, 5.0, 0.5, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 0D arrays (scalar-like)
    x = np.array(0.0, dtype=np.float32)
    y = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: Broadcasting: 1D 'x' and 2D 'y'
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.random.uniform(0.1, 5.0, size=(2, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: Broadcasting: 3D 'x' and 1D 'y'
    x = np.random.uniform(0.1, 5.0, size=(2, 2, 3)).astype(np.float32)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: Negative 'x', positive 'y'
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: Large values
    x = np.random.uniform(100.0, 1000.0, size=(5,)).astype(np.float32)
    y = np.random.uniform(100.0, 1000.0, size=(5,)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: Small positive values for 'y'
    x = np.array([1.0, 2.0], dtype=np.float32)
    y = np.array([1e-5, 1e-6], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: Higher dimensional arrays (4D)
    x = np.random.uniform(0.1, 2.0, size=(2, 2, 2, 2)).astype(np.float32)
    y = np.random.uniform(0.1, 2.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs

generated_inputs["jax.scipy.special.xlogy"] = xlogy_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.xlogy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.xlogy'.")


check_valid('jax.scipy.special.xlogy', generated_inputs['jax.scipy.special.xlogy'], lib="jax", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def xlogy_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays with positive values
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 4.0, 8.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: Testing x=0 boundary condition, 1D float32
    x = np.array([0.0, 0.0, 1.5, 0.0], dtype=np.float32)
    y = np.array([0.0, 2.0, 3.0, 10.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 3: 2D float64 arrays with positive values
    x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    y = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 4: Broadcasting shapes (3, 1) and (1, 4)
    x = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([[0.5, 1.5, 2.5, 3.5]], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 5: 3D float32 arrays
    x = np.random.uniform(0.1, 5.0, size=(2, 3, 4)).astype(np.float32)
    y = np.random.uniform(0.1, 5.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 6: Scalar-like 0D arrays (shape ())
    x = np.array(0.0, dtype=np.float32)
    y = np.array(0.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 7: Large positive float64 values
    x = np.array([1e4, 2.5e5], dtype=np.float64)
    y = np.array([1e2, 5e3], dtype=np.float64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 8: Negative x values, positive y values
    x = np.array([-1.0, -2.5, -0.5], dtype=np.float32)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 9: Small fractional float32 values
    x = np.array([1e-5, 2e-5, 3e-5], dtype=np.float32)
    y = np.array([1e-3, 2e-3, 3e-3], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 10: 1D float64 array with both positive, zero, and broadcasting
    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    y = np.array([5.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "y": y})

    return list_of_inputs

generated_inputs["jax.scipy.special.xlogy_1"] = xlogy_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.xlogy_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.xlogy_1'.")


check_valid('jax.scipy.special.xlogy', generated_inputs['jax.scipy.special.xlogy_1'], lib="jax", suffix=1)

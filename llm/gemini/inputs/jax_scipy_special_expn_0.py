
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def expn_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays, integer n and float32 x
    n = np.array([1, 2, 3], dtype=np.int32)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"n": n, "x": x})

    # Input 2: Scalar values (0-D arrays)
    n = np.array(1, dtype=np.int32)
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"n": n, "x": x})

    # Input 3: n as float, x as float32
    n = np.array([1.5, 2.5], dtype=np.float32)
    x = np.array([0.1, 2.0], dtype=np.float32)
    list_of_inputs.append({"n": n, "x": x})

    # Input 4: float64 values
    n = np.array([0, 1, 2, 3], dtype=np.float64)
    x = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    list_of_inputs.append({"n": n, "x": x})

    # Input 5: 2D arrays with matching shapes
    n = np.array([[1, 2], [3, 4]], dtype=np.int32)
    x = np.array([[1.0, 1.5], [2.0, 2.5]], dtype=np.float32)
    list_of_inputs.append({"n": n, "x": x})

    # Input 6: Broadcasting shapes (3, 1) and (1, 4)
    n = np.array([[1], [2], [3]], dtype=np.int32)
    x = np.array([[0.5, 1.0, 1.5, 2.0]], dtype=np.float32)
    list_of_inputs.append({"n": n, "x": x})

    # Input 7: Large x values
    n = np.array([1, 2], dtype=np.int32)
    x = np.array([10.0, 20.0], dtype=np.float32)
    list_of_inputs.append({"n": n, "x": x})

    # Input 8: Negative/zero n values with positive x
    n = np.array([-1, 0, 1], dtype=np.int32)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"n": n, "x": x})

    # Input 9: Small x values close to zero
    n = np.array([1, 1], dtype=np.int32)
    x = np.array([1e-3, 1e-2], dtype=np.float32)
    list_of_inputs.append({"n": n, "x": x})

    # Input 10: 3D arrays
    n = np.ones((2, 2, 2), dtype=np.int32) * 2
    x = np.random.uniform(0.5, 5.0, size=(2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"n": n, "x": x})

    return list_of_inputs

generated_inputs["jax.scipy.special.expn"] = expn_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.expn' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.expn'.")


check_valid('jax.scipy.special.expn', generated_inputs['jax.scipy.special.expn'], lib="jax", suffix=0)

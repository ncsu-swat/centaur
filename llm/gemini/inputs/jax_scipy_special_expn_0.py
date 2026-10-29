
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def expn_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars
    n = np.array(1, dtype=np.int32)
    x = np.array(2.0, dtype=np.float32)
    list_of_inputs.append({"n": n, "x": x})

    # Input 2: 1D arrays of matching shape
    n = np.array([0, 1, 2, 3], dtype=np.int32)
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append({"n": n, "x": x})

    # Input 3: 2D arrays float64
    n = np.array([[1, 2], [3, 4]], dtype=np.int64)
    x = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    list_of_inputs.append({"n": n, "x": x})

    # Input 4: Broadcasting 2D with 1D
    n = np.array([[1], [2], [3]], dtype=np.int32)
    x = np.array([0.1, 1.0, 5.0], dtype=np.float32)
    list_of_inputs.append({"n": n, "x": x})

    # Input 5: 3D arrays
    n = np.zeros((2, 3, 4), dtype=np.int32)
    x = np.ones((2, 3, 4), dtype=np.float32)
    list_of_inputs.append({"n": n, "x": x})

    # Input 6: Floating point n
    n = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    x = np.array([0.5, 1.2, 2.3], dtype=np.float32)
    list_of_inputs.append({"n": n, "x": x})

    # Input 7: Large x values with float64
    n = np.array([1, 2, 3], dtype=np.int32)
    x = np.array([10.0, 25.0, 50.0], dtype=np.float64)
    list_of_inputs.append({"n": n, "x": x})

    # Input 8: Small positive x values
    n = np.array([2, 3, 4], dtype=np.int32)
    x = np.array([1e-3, 1e-2, 1e-1], dtype=np.float32)
    list_of_inputs.append({"n": n, "x": x})

    # Input 9: 4D arrays
    n = np.ones((2, 1, 3, 2), dtype=np.int32)
    x = np.full((2, 2, 3, 2), 2.5, dtype=np.float32)
    list_of_inputs.append({"n": n, "x": x})

    # Input 10: High order n
    n = np.array([5, 10, 20, 50], dtype=np.int32)
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append({"n": n, "x": x})

    # Input 11: Scalar n broadcasting with 1D array x
    n = np.array(2, dtype=np.int32)
    x = np.linspace(0.1, 10.0, 8, dtype=np.float64)
    list_of_inputs.append({"n": n, "x": x})

    return [copy.deepcopy(inp) for inp in list_of_inputs]


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

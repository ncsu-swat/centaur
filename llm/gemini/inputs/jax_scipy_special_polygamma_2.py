
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def polygamma_inputs():
    list_of_inputs = []

    # Input 1: Scalar inputs
    n = np.array(0, dtype=np.int32)
    x = np.array(1.0, dtype=np.float32)
    list_of_inputs.append({"n": n, "x": x})

    # Input 2: 1D arrays of the same shape
    n = np.array([1, 2, 3], dtype=np.int32)
    x = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    list_of_inputs.append({"n": n, "x": x})

    # Input 3: 2D arrays of the same shape
    n = np.array([[0, 1], [2, 3]], dtype=np.int32)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append({"n": n, "x": x})

    # Input 4: Broadcasting shapes (2, 1) and (3,)
    n = np.array([[1], [2]], dtype=np.int32)
    x = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    list_of_inputs.append({"n": n, "x": x})

    # Input 5: float64 and int64
    n = np.array([0, 1, 2], dtype=np.int64)
    x = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    list_of_inputs.append({"n": n, "x": x})

    # Input 6: Broadcasting scalar n to x
    n = np.array(4, dtype=np.int32)
    x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append({"n": n, "x": x})

    # Input 7: Random values for x, zeros for n
    n = np.zeros((3, 3), dtype=np.int32)
    x = np.random.uniform(0.1, 10.0, size=(3, 3)).astype(np.float32)
    list_of_inputs.append({"n": n, "x": x})

    # Input 8: 3D tensors
    n = np.ones((2, 2, 2), dtype=np.int32)
    x = np.random.uniform(1.0, 5.0, size=(2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"n": n, "x": x})

    # Input 9: Large x values, int16 for n
    n = np.array([0, 1, 2, 3, 4], dtype=np.int16)
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
    list_of_inputs.append({"n": n, "x": x})

    # Input 10: x values close to zero (but positive)
    n = np.array([2, 2], dtype=np.int32)
    x = np.array([0.1, 0.9], dtype=np.float32)
    list_of_inputs.append({"n": n, "x": x})

    return list_of_inputs

generated_inputs["jax.scipy.special.polygamma_2"] = polygamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.polygamma_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.polygamma_2'.")


check_valid('jax.scipy.special.polygamma', generated_inputs['jax.scipy.special.polygamma_2'], lib="jax", suffix=2)

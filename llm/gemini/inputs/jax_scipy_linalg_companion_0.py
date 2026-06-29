
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def companion_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, length 4
    a1 = np.array([1.0, -10.0, 31.0, -30.0], dtype=np.float32)
    list_of_inputs.append({"a": a1})

    # Input 2: 1D float64, length 2 (minimum size)
    a2 = np.array([2.0, -5.0], dtype=np.float64)
    list_of_inputs.append({"a": a2})

    # Input 3: 1D float32, larger size
    a3 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    list_of_inputs.append({"a": a3})

    # Input 4: 2D float32, shape (3, 4) - batch of companion matrices
    a4 = np.random.randn(3, 4).astype(np.float32)
    a4[:, 0] = np.random.uniform(1.0, 2.0, size=(3,))  # Ensure leading coeffs are non-zero
    list_of_inputs.append({"a": a4})

    # Input 5: 2D float64, shape (2, 3)
    a5 = np.random.randn(2, 3).astype(np.float64)
    a5[:, 0] = 1.5
    list_of_inputs.append({"a": a5})

    # Input 6: 3D float32, shape (2, 2, 4)
    a6 = np.random.randn(2, 2, 4).astype(np.float32)
    a6[..., 0] = np.random.uniform(0.5, 1.5, size=(2, 2))
    list_of_inputs.append({"a": a6})

    # Input 7: 1D float32, negative elements
    a7 = np.array([-3.0, -2.0, -1.0, 0.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"a": a7})

    # Input 8: 2D float32, shape (4, 5) with negative leading elements
    a8 = np.random.randn(4, 5).astype(np.float32)
    a8[:, 0] = -2.5
    list_of_inputs.append({"a": a8})

    # Input 9: 3D float64, shape (1, 3, 5)
    a9 = np.random.randn(1, 3, 5).astype(np.float64)
    a9[..., 0] = 1.0
    list_of_inputs.append({"a": a9})

    # Input 10: 1D float32, size 8, leading non-zero
    a10 = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    list_of_inputs.append({"a": a10})

    return list_of_inputs

generated_inputs["jax.scipy.linalg.companion"] = companion_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.companion' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.companion'.")


check_valid('jax.scipy.linalg.companion', generated_inputs['jax.scipy.linalg.companion'], lib="jax", suffix=0)

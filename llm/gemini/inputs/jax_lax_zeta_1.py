
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def zeta_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, float32, standard values
    x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    q = np.array([1.0, 1.5, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "q": q})

    # Input 2: 2D arrays, float32
    x = np.array([[2.5, 3.5], [4.5, 5.5]], dtype=np.float32)
    q = np.array([[0.5, 1.2], [2.3, 3.1]], dtype=np.float32)
    list_of_inputs.append({"x": x, "q": q})

    # Input 3: 0D arrays (scalars)
    x = np.array(2.0, dtype=np.float32)
    q = np.array(1.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "q": q})

    # Input 4: float64 arrays, 1D
    x = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    q = np.array([0.1, 0.5, 1.0, 2.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "q": q})

    # Input 5: Broadcastable shapes (3, 1) and (1, 4)
    x = np.array([[2.0], [3.0], [4.0]], dtype=np.float32)
    q = np.array([[1.0, 1.5, 2.0, 2.5]], dtype=np.float32)
    list_of_inputs.append({"x": x, "q": q})

    # Input 6: 3D arrays, float32
    x = np.random.uniform(1.5, 5.0, size=(2, 2, 2)).astype(np.float32)
    q = np.random.uniform(0.5, 3.0, size=(2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": x, "q": q})

    # Input 7: Large values of x and q
    x = np.array([10.0, 20.0, 50.0], dtype=np.float32)
    q = np.array([5.0, 10.0, 20.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "q": q})

    # Input 8: Close to boundary values (x slightly > 1, q slightly > 0)
    x = np.array([1.01, 1.05, 1.1], dtype=np.float64)
    q = np.array([0.01, 0.05, 0.1], dtype=np.float64)
    list_of_inputs.append({"x": x, "q": q})

    # Input 9: Broadcastable shapes (1, 5) and (5, 5)
    x = np.random.uniform(2.0, 4.0, size=(1, 5)).astype(np.float32)
    q = np.random.uniform(1.0, 2.0, size=(5, 5)).astype(np.float32)
    list_of_inputs.append({"x": x, "q": q})

    # Input 10: 4D arrays, float32
    x = np.random.uniform(1.2, 3.0, size=(2, 2, 2, 2)).astype(np.float32)
    q = np.random.uniform(0.8, 2.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": x, "q": q})

    return list_of_inputs

generated_inputs["jax.lax.zeta_1"] = zeta_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.zeta_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.zeta_1'.")


check_valid('jax.lax.zeta', generated_inputs['jax.lax.zeta_1'], lib="jax", suffix=1)

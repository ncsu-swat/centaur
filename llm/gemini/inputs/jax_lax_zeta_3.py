
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_zeta_inputs():
    list_of_inputs = []

    # Input 1: Scalar inputs (0D arrays), float32 with integer values
    x = np.array(2.0, dtype=np.float32)
    q = np.array(1.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "q": copy.deepcopy(q)})

    # Input 2: Scalar inputs, float64 with integer values
    x = np.array(3.0, dtype=np.float64)
    q = np.array(2.0, dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "q": copy.deepcopy(q)})

    # Input 3: 1D arrays, size 5, float32 with integer values
    x = np.array([2.0, 3.0, 4.0, 2.0, 3.0], dtype=np.float32)
    q = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "q": copy.deepcopy(q)})

    # Input 4: 2D arrays, 3x3, float32 with integer values
    x = np.random.randint(2, 10, size=(3, 3)).astype(np.float32)
    q = np.random.randint(1, 10, size=(3, 3)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "q": copy.deepcopy(q)})

    # Input 5: 3D arrays, 2x2x2, float64 with integer values
    x = np.random.randint(2, 6, size=(2, 2, 2)).astype(np.float64)
    q = np.random.randint(1, 6, size=(2, 2, 2)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "q": copy.deepcopy(q)})

    # Input 6: Broadcastable shapes (3, 1) and (1, 3), float32 with integer values
    x = np.array([[2.0], [3.0], [4.0]], dtype=np.float32)
    q = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "q": copy.deepcopy(q)})

    # Input 7: Scalar x and 1D array q, float32 with integer values
    x = np.array(2.0, dtype=np.float32)
    q = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "q": copy.deepcopy(q)})

    # Input 8: 2D array x and scalar q, float32 with integer values
    x = np.random.randint(2, 6, size=(2, 4)).astype(np.float32)
    q = np.array(3.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "q": copy.deepcopy(q)})

    # Input 9: 1D arrays of size 10, float64 with integer values
    x = np.random.randint(2, 8, size=(10,)).astype(np.float64)
    q = np.random.randint(1, 8, size=(10,)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "q": copy.deepcopy(q)})

    # Input 10: 4D arrays, 2x2x2x2, float32 with integer values
    x = np.random.randint(2, 5, size=(2, 2, 2, 2)).astype(np.float32)
    q = np.random.randint(1, 5, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "q": copy.deepcopy(q)})

    return list_of_inputs

generated_inputs["jax.lax.zeta_3"] = jax_lax_zeta_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.zeta_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.zeta_3'.")


check_valid('jax.lax.zeta', generated_inputs['jax.lax.zeta_3'], lib="jax", suffix=3)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def kl_div_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, positive float32
    p = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    q = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"p": p, "q": q})

    # Input 2: 2D arrays, positive float64
    p = np.random.uniform(0.1, 10.0, size=(3, 3)).astype(np.float64)
    q = np.random.uniform(0.1, 10.0, size=(3, 3)).astype(np.float64)
    list_of_inputs.append({"p": p, "q": q})

    # Input 3: 1D arrays, including zeros, float32
    p = np.array([0.0, 1.0, 0.0, 2.5], dtype=np.float32)
    q = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append({"p": p, "q": q})

    # Input 4: 3D arrays, random positives, float32
    p = np.random.uniform(0.5, 5.0, size=(2, 3, 4)).astype(np.float32)
    q = np.random.uniform(0.5, 5.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"p": p, "q": q})

    # Input 5: 1D arrays, including negative values, float32
    p = np.array([-1.0, 0.0, 2.0], dtype=np.float32)
    q = np.array([2.0, -3.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"p": p, "q": q})

    # Input 6: 0D arrays (scalars wrapped in np.array), float32
    p = np.array(1.5, dtype=np.float32)
    q = np.array(2.5, dtype=np.float32)
    list_of_inputs.append({"p": p, "q": q})

    # Input 7: Broadcasting test - 2D and 1D
    p = np.random.uniform(0.1, 5.0, size=(3, 4)).astype(np.float32)
    q = np.random.uniform(0.1, 5.0, size=(4,)).astype(np.float32)
    list_of_inputs.append({"p": p, "q": q})

    # Input 8: Broadcasting test - 1D and 2D
    p = np.random.uniform(0.1, 5.0, size=(5, 1)).astype(np.float64)
    q = np.random.uniform(0.1, 5.0, size=(1, 5)).astype(np.float64)
    list_of_inputs.append({"p": p, "q": q})

    # Input 9: Large 2D arrays, float32
    p = np.random.exponential(scale=1.0, size=(100, 100)).astype(np.float32)
    q = np.random.exponential(scale=1.0, size=(100, 100)).astype(np.float32)
    list_of_inputs.append({"p": p, "q": q})

    # Input 10: 4D arrays, float32
    p = np.random.uniform(0.1, 2.0, size=(2, 2, 3, 3)).astype(np.float32)
    q = np.random.uniform(0.1, 2.0, size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append({"p": p, "q": q})

    return list_of_inputs

generated_inputs["jax.scipy.special.kl_div"] = kl_div_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.kl_div' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.kl_div'.")


check_valid('jax.scipy.special.kl_div', generated_inputs['jax.scipy.special.kl_div'], lib="jax", suffix=0)

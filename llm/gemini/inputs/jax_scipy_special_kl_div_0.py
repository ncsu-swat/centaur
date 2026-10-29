
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def kl_div_inputs():
    list_of_inputs = []

    # Input 1: 1D positive float32 arrays
    p = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    q = np.array([0.2, 0.4, 0.8], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"p": p, "q": q}))

    # Input 2: 2D positive float64 arrays
    p = np.array([[0.25, 0.75], [0.1, 0.9]], dtype=np.float64)
    q = np.array([[0.5, 0.5], [0.3, 0.7]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"p": p, "q": q}))

    # Input 3: 0D scalar arrays
    p = np.array(0.5, dtype=np.float32)
    q = np.array(0.2, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"p": p, "q": q}))

    # Input 4: Broadcasting shapes (3, 1) and (1, 4)
    p = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    q = np.array([[0.5, 1.0, 1.5, 2.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"p": p, "q": q}))

    # Input 5: Containing zeros (p=0, q>=0)
    p = np.array([0.0, 0.2, 0.0, 0.5], dtype=np.float32)
    q = np.array([1.0, 0.2, 0.0, 0.1], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"p": p, "q": q}))

    # Input 6: Containing negative values
    p = np.array([-1.0, 0.5, -0.2], dtype=np.float32)
    q = np.array([0.5, -0.5, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"p": p, "q": q}))

    # Input 7: 3D float32 arrays
    p = np.random.uniform(0.1, 10.0, size=(2, 3, 4)).astype(np.float32)
    q = np.random.uniform(0.1, 10.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"p": p, "q": q}))

    # Input 8: Integer arrays
    p = np.array([1, 2, 3, 4], dtype=np.int32)
    q = np.array([2, 2, 2, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"p": p, "q": q}))

    # Input 9: Identical arrays (kl_div should be 0 where p == q > 0)
    p = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    q = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"p": p, "q": q}))

    # Input 10: 4D arrays with broadcasting along leading dimensions
    p = np.random.uniform(0.01, 1.0, size=(2, 1, 4, 5)).astype(np.float32)
    q = np.random.uniform(0.01, 1.0, size=(1, 3, 4, 5)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"p": p, "q": q}))

    # Input 11: 1D arrays with float16
    p = np.array([0.2, 0.4, 0.6], dtype=np.float16)
    q = np.array([0.1, 0.5, 0.9], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"p": p, "q": q}))

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


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def rel_entr_inputs():
    list_of_inputs = []

    # Input 1: 1D positive float32 arrays
    p = np.array([0.5, 0.2, 0.3], dtype=np.float32)
    q = np.array([0.4, 0.3, 0.3], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"p": p, "q": q}))

    # Input 2: 2D positive float64 arrays
    p = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    q = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"p": p, "q": q}))

    # Input 3: 0D scalar arrays
    p = np.array(0.5, dtype=np.float32)
    q = np.array(0.2, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"p": p, "q": q}))

    # Input 4: Inputs containing zeros (valid edge cases per definition)
    p = np.array([0.0, 0.5, 0.0], dtype=np.float32)
    q = np.array([0.2, 0.5, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"p": p, "q": q}))

    # Input 5: Inputs with negative values (evaluates to infinity per definition)
    p = np.array([-1.0, 0.5, 2.0], dtype=np.float32)
    q = np.array([1.0, -0.5, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"p": p, "q": q}))

    # Input 6: Broadcasting shapes (3, 1) and (1, 4)
    p = np.linspace(0.1, 1.0, 3, dtype=np.float32).reshape(3, 1)
    q = np.linspace(0.2, 0.8, 4, dtype=np.float32).reshape(1, 4)
    list_of_inputs.append(copy.deepcopy({"p": p, "q": q}))

    # Input 7: 3D arrays
    p = np.ones((2, 3, 4), dtype=np.float64) * 0.25
    q = np.ones((2, 3, 4), dtype=np.float64) * 0.5
    list_of_inputs.append(copy.deepcopy({"p": p, "q": q}))

    # Input 8: Integer dtypes (real-valued)
    p = np.array([1, 2, 3, 0], dtype=np.int32)
    q = np.array([2, 2, 1, 1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"p": p, "q": q}))

    # Input 9: float16 arrays
    p = np.array([0.1, 0.4, 0.9], dtype=np.float16)
    q = np.array([0.2, 0.3, 0.8], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"p": p, "q": q}))

    # Input 10: 4D tensors with broadcasting
    p = np.random.uniform(0.1, 1.0, size=(2, 1, 4, 5)).astype(np.float32)
    q = np.random.uniform(0.1, 1.0, size=(1, 3, 4, 5)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"p": p, "q": q}))

    return list_of_inputs


generated_inputs["jax.scipy.special.rel_entr"] = rel_entr_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.rel_entr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.rel_entr'.")


check_valid('jax.scipy.special.rel_entr', generated_inputs['jax.scipy.special.rel_entr'], lib="jax", suffix=0)

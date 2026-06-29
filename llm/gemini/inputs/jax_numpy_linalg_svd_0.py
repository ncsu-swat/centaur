
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def svd_inputs():
    list_of_inputs = []

    # Keeping all static parameters, shapes, and the subset_by_index tuple identical 
    # to compile exactly ONCE and satisfy the strict 'tuple' type requirement.
    for _ in range(10):
        a = np.random.randn(2, 2).astype(np.float32)
        input_dict = {
            "a": a,
            "full_matrices": False,
            "compute_uv": True,
            "hermitian": False,
            "subset_by_index": (0, 1)
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.svd"] = svd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.svd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.svd'.")


check_valid('jax.numpy.linalg.svd', generated_inputs['jax.numpy.linalg.svd'], lib="jax", suffix=0)

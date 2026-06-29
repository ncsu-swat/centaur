
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def vq_inputs():
    list_of_inputs = []

    # 10 identical minimal 1D inputs to minimize execution/compilation overhead per run
    for _ in range(10):
        obs = np.array([1.0], dtype=np.float32)
        code_book = np.array([1.0], dtype=np.float32)
        list_of_inputs.append({
            "obs": obs,
            "code_book": code_book,
            "check_finite": False
        })

    return list_of_inputs

generated_inputs["jax.scipy.cluster.vq.vq"] = vq_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.cluster.vq.vq' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.cluster.vq.vq'.")


check_valid('jax.scipy.cluster.vq.vq', generated_inputs['jax.scipy.cluster.vq.vq'], lib="jax", suffix=0)

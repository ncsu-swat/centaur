
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np

def gamma_inputs():
    list_of_inputs = []
    
    list_of_inputs.append({"x": np.array([1.0], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([2.0], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([3.0], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([4.0], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([5.0], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([1.5], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([2.5], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([3.5], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([4.5], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([5.5], dtype=np.float32)})
    
    return list_of_inputs

generated_inputs["jax.scipy.special.gamma"] = gamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.gamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.gamma'.")


check_valid('jax.scipy.special.gamma', generated_inputs['jax.scipy.special.gamma'], lib="jax", suffix=0)

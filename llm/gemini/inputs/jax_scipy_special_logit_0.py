
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def logit_inputs():
    list_of_inputs = []

    # Input 1: 0D array (scalar) float32
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D array float32
    x = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D array float64 with random values between 0 and 1
    x = np.random.uniform(0.01, 0.99, size=(3, 3)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D array float32
    x = np.random.uniform(0.1, 0.9, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D array float16
    x = np.array([0.25, 0.5, 0.75], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D array with extreme values near 0 and 1 (float64)
    x = np.array([1e-6, 0.5, 1.0 - 1e-6], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 4D array float32
    x = np.random.uniform(0.2, 0.8, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 2D array including boundary values 0.0 and 1.0
    x = np.array([[0.0, 0.5], [0.5, 1.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D array with single element float64
    x = np.array([0.12345], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Large 1D array float32
    x = np.random.uniform(0.01, 0.99, size=(100,)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.scipy.special.logit"] = logit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.logit' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.logit'.")


check_valid('jax.scipy.special.logit', generated_inputs['jax.scipy.special.logit'], lib="jax", suffix=0)

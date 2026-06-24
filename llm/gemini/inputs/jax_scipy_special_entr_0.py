
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def entr_inputs():
    list_of_inputs = []

    # Input 1: 1D array with positive float32 values
    x = np.array([0.1, 0.5, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 2: 2D array with positive and zero values (float32)
    x = np.array([[0.0, 0.5], [1.5, 2.5]], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 3: 1D array with negative values (float32)
    x = np.array([-1.0, -0.5, -0.1], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 4: 3D array with positive float64 values
    x = np.random.uniform(0.1, 10.0, size=(2, 3, 4)).astype(np.float64)
    list_of_inputs.append({"x": x})

    # Input 5: Scalar (0D array) positive value
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 6: Scalar (0D array) negative value
    x = np.array(-0.5, dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 7: 1D array with float16 type
    x = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float16)
    list_of_inputs.append({"x": x})

    # Input 8: 4D array with mixed values (float32)
    x = np.array([[[[-0.1, 0.0], [0.5, 1.0]]]], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 9: Large 2D float32 array
    x = np.random.uniform(0.01, 5.0, size=(10, 10)).astype(np.float32)
    list_of_inputs.append({"x": x})

    # Input 10: 1D array with extremely small values
    x = np.array([1e-10, 1e-20, 1e-30], dtype=np.float64)
    list_of_inputs.append({"x": x})

    # Input 11: 1D array with some extremely large values
    x = np.array([1e5, 1e10, 1e20], dtype=np.float32)
    list_of_inputs.append({"x": x})

    return list_of_inputs

generated_inputs["jax.scipy.special.entr"] = entr_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.entr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.entr'.")


check_valid('jax.scipy.special.entr', generated_inputs['jax.scipy.special.entr'], lib="jax", suffix=0)

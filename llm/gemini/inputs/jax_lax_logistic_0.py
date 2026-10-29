
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def logistic_inputs():
    list_of_inputs = []

    # Input 1: 1-D float32 array with positive, negative, and zero values
    x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=np.float32)
    accuracy = "DEFAULT"
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 2: 2-D float64 matrix
    x = np.random.randn(3, 4).astype(np.float64)
    accuracy = "HIGH"
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 3: 0-D scalar float32
    x = np.array(0.5, dtype=np.float32)
    accuracy = "HIGHEST"
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 4: 3-D float32 array
    x = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    accuracy = "TOLERANT"
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 5: 1-D float16 array
    x = np.array([-2.5, -0.5, 0.0, 0.5, 2.5], dtype=np.float16)
    accuracy = "default"
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 6: 4-D float32 array with extreme values (saturation test)
    x = np.array([[[[-100.0, -50.0], [50.0, 100.0]]]], dtype=np.float32)
    accuracy = "high"
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 7: 2-D complex64 array
    x = (np.random.randn(2, 3) + 1j * np.random.randn(2, 3)).astype(np.complex64)
    accuracy = "DEFAULT"
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 8: 1-D complex128 array
    x = np.array([1.0 + 2.0j, -1.0 - 0.5j, 0.0 + 0.0j], dtype=np.complex128)
    accuracy = "highest"
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 9: 5-D float64 array
    x = np.random.randn(1, 2, 1, 3, 2).astype(np.float64)
    accuracy = "tolerant"
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 10: 2-D float32 array with values close to zero
    x = np.array([[-1e-5, 0.0], [1e-5, -1e-6]], dtype=np.float32)
    accuracy = "DEFAULT"
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 11: 3-D float16 array
    x = np.random.randn(2, 2, 2).astype(np.float16)
    accuracy = "HIGH"
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    return list_of_inputs


generated_inputs["jax.lax.logistic"] = logistic_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.logistic' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.logistic'.")


check_valid('jax.lax.logistic', generated_inputs['jax.lax.logistic'], lib="jax", suffix=0)

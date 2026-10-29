
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def atanh_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    x = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D single-element float32 array
    x = np.array([0.25], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D float32 array
    x = np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D float64 array near boundaries
    x = np.array([-0.99, 0.0, 0.99], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D float32 array
    x = np.random.uniform(-0.8, 0.8, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D complex64 array
    x = np.array([0.5 + 0.2j, -0.3 - 0.4j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 2D complex128 array
    x = np.array([[0.1 + 0.1j, -0.5 + 0.2j], [0.0 + 0.0j, 0.8 - 0.1j]], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D array of negative values
    x = np.array([-0.8, -0.5, -0.2, -0.01], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 4D float32 array
    x = np.random.uniform(-0.7, 0.7, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 2D float64 array
    x = np.array([[-0.75, 0.25, 0.0], [0.5, -0.5, 0.1]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D array of zeros
    x = np.zeros((5,), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs

generated_inputs["jax.numpy.atanh_5"] = atanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.atanh_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.atanh_5'.")


check_valid('jax.numpy.atanh', generated_inputs['jax.numpy.atanh_5'], lib="jax", suffix=5)

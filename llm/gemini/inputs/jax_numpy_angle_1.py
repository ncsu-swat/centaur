
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def angle_inputs():
    list_of_inputs = []

    # Input 1: 1D complex64 array, radians
    z = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64)
    deg = False
    list_of_inputs.append({"z": copy.deepcopy(z), "deg": deg})

    # Input 2: 2D complex128 array, degrees
    z = np.array([[1+2j, -3+4j], [5-6j, -7-8j]], dtype=np.complex128)
    deg = True
    list_of_inputs.append({"z": copy.deepcopy(z), "deg": deg})

    # Input 3: 3D complex64 array, radians
    z = np.random.randn(2, 3, 4).astype(np.float32) + 1j * np.random.randn(2, 3, 4).astype(np.float32)
    deg = False
    list_of_inputs.append({"z": copy.deepcopy(z), "deg": deg})

    # Input 4: 1D float32 array (real values), degrees
    z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    deg = True
    list_of_inputs.append({"z": copy.deepcopy(z), "deg": deg})

    # Input 5: 2D float64 array, radians
    z = np.random.randn(5, 5).astype(np.float64)
    deg = False
    list_of_inputs.append({"z": copy.deepcopy(z), "deg": deg})

    # Input 6: 4D complex128 array, degrees
    z = (np.random.randn(2, 2, 2, 2) + 1j * np.random.randn(2, 2, 2, 2)).astype(np.complex128)
    deg = True
    list_of_inputs.append({"z": copy.deepcopy(z), "deg": deg})

    # Input 7: 0D complex64 array (scalar-like), radians
    z = np.array(3 - 4j, dtype=np.complex64)
    deg = False
    list_of_inputs.append({"z": copy.deepcopy(z), "deg": deg})

    # Input 8: 1D purely imaginary complex64, degrees
    z = np.array([1j, 2j, -3j, -4j], dtype=np.complex64)
    deg = True
    list_of_inputs.append({"z": copy.deepcopy(z), "deg": deg})

    # Input 9: 2D mixed array, radians
    z = np.array([[0j, 1+0j], [-1+0j, 0-1j]], dtype=np.complex128)
    deg = False
    list_of_inputs.append({"z": copy.deepcopy(z), "deg": deg})

    # Input 10: 3D float32, degrees
    z = np.ones((2, 2, 2), dtype=np.float32) * -1.5
    deg = True
    list_of_inputs.append({"z": copy.deepcopy(z), "deg": deg})

    return list_of_inputs

generated_inputs["jax.numpy.angle_1"] = angle_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.angle_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.angle_1'.")


check_valid('jax.numpy.angle', generated_inputs['jax.numpy.angle_1'], lib="jax", suffix=1)

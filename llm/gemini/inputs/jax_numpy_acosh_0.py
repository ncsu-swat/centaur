
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def acosh_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, values >= 1
    x = np.random.uniform(1.0, 10.0, size=(5,)).astype(np.float32)
    list_of_inputs.append({"x": x})

    # Input 2: 2D array, float64, values >= 1
    x = np.random.uniform(1.0, 100.0, size=(3, 4)).astype(np.float64)
    list_of_inputs.append({"x": x})

    # Input 3: 3D array, float32, some values < 1
    x = np.random.uniform(-5.0, 5.0, size=(2, 2, 3)).astype(np.float32)
    list_of_inputs.append({"x": x})

    # Input 4: 1D array, complex64
    real_part = np.random.uniform(-5.0, 5.0, size=(5,))
    imag_part = np.random.uniform(-5.0, 5.0, size=(5,))
    x = (real_part + 1j * imag_part).astype(np.complex64)
    list_of_inputs.append({"x": x})

    # Input 5: 0D array (scalar), float32 >= 1
    x = np.array(2.5, dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 6: 4D array, float16, values >= 1
    x = np.random.uniform(1.0, 5.0, size=(2, 2, 2, 2)).astype(np.float16)
    list_of_inputs.append({"x": x})

    # Input 7: 2D array, int32, values >= 1
    x = np.random.randint(1, 100, size=(3, 3)).astype(np.int32)
    list_of_inputs.append({"x": x})

    # Input 8: 1D array, float64, very large values
    x = np.random.uniform(100.0, 10000.0, size=(10,)).astype(np.float64)
    list_of_inputs.append({"x": x})

    # Input 9: 3D array, complex128
    real_part = np.random.uniform(-10.0, 10.0, size=(2, 3, 2))
    imag_part = np.random.uniform(-10.0, 10.0, size=(2, 3, 2))
    x = (real_part + 1j * imag_part).astype(np.complex128)
    list_of_inputs.append({"x": x})

    # Input 10: 5D array, float32, values >= 1
    x = np.random.uniform(1.0, 2.0, size=(1, 2, 1, 3, 2)).astype(np.float32)
    list_of_inputs.append({"x": x})

    return list_of_inputs

generated_inputs["jax.numpy.acosh"] = acosh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.acosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.acosh'.")


check_valid('jax.numpy.acosh', generated_inputs['jax.numpy.acosh'], lib="jax", suffix=0)

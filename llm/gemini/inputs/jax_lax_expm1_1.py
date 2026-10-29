
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def expm1_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with values near zero
    x = np.array([-1e-5, -1e-7, 0.0, 1e-7, 1e-5], dtype=np.float32)
    accuracy = ""
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 2: 2D float64 matrix with mixed positive and negative values
    x = np.random.randn(3, 4).astype(np.float64)
    accuracy = ""
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 3: 0D scalar array float32
    x = np.array(0.5, dtype=np.float32)
    accuracy = ""
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 4: 1D float16 array
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float16)
    accuracy = ""
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 5: 3D float32 tensor with strictly negative values
    x = -np.random.uniform(0.1, 10.0, size=(2, 3, 4)).astype(np.float32)
    accuracy = ""
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 6: 2D complex64 array
    real_part = np.random.randn(2, 3).astype(np.float32)
    imag_part = np.random.randn(2, 3).astype(np.float32)
    x = (real_part + 1j * imag_part).astype(np.complex64)
    accuracy = ""
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 7: 1D complex128 array
    real_part = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    imag_part = np.array([0.5, -0.5, 2.0], dtype=np.float64)
    x = (real_part + 1j * imag_part).astype(np.complex128)
    accuracy = ""
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 8: 4D float32 tensor
    x = np.random.uniform(-1.0, 1.0, size=(2, 2, 3, 2)).astype(np.float32)
    accuracy = ""
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 9: 2D float64 array with large negative values
    x = np.array([[-100.0, -50.0], [-20.0, -10.0]], dtype=np.float64)
    accuracy = ""
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 10: 1D float32 array with positive values
    x = np.linspace(0.1, 5.0, 10, dtype=np.float32)
    accuracy = ""
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 11: 3D float64 array with small values around zero
    x = np.random.uniform(-0.01, 0.01, size=(3, 2, 2)).astype(np.float64)
    accuracy = ""
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    return list_of_inputs


generated_inputs["jax.lax.expm1_1"] = expm1_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.expm1_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.expm1_1'.")


check_valid('jax.lax.expm1', generated_inputs['jax.lax.expm1_1'], lib="jax", suffix=1)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def cos_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, small positive and negative values
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32, standard matrix
    x = np.random.randn(3, 4).astype(np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64, higher precision
    x = np.random.randn(2, 2, 3).astype(np.float64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D complex64, complex numbers support
    x = np.array([1.0 + 1.0j, -2.0 + 0.5j, 0.0 - 3.0j], dtype=np.complex64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D complex128, high precision complex
    real = np.random.randn(2, 3)
    imag = np.random.randn(2, 3)
    x = (real + 1j * imag).astype(np.complex128)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32, higher dimensions
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 0D float32 (scalar)
    x = np.array(1.5, dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float16, lower precision floating point
    x = np.linspace(-np.pi, np.pi, 10).astype(np.float16)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32, large array
    x = np.random.uniform(-10.0, 10.0, size=(50, 50)).astype(np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float32, containing zeros
    x = np.zeros((3, 3, 3), dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.cos_1"] = cos_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.cos_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.cos_1'.")


check_valid('jax.lax.cos', generated_inputs['jax.lax.cos_1'], lib="jax", suffix=1)

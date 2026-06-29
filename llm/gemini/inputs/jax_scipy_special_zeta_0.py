
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def zeta_inputs():
    list_of_inputs = []

    # Input 1: 0D arrays (scalars), float32
    x = np.array(2.5, dtype=np.float32)
    q = np.array(1.5, dtype=np.float32)
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays, float32
    x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    q = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D arrays, float64
    x = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    q = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays, float32
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    q = np.array([[1.0, 1.5], [2.0, 2.5]], dtype=np.float32)
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D arrays, float64
    x = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64)
    q = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting (3, 1) and (1, 3), float32
    x = np.array([[2.0], [3.0], [4.0]], dtype=np.float32)
    q = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcasting (1, 2) and (2, 1), float64
    x = np.array([[1.5, 2.5]], dtype=np.float64)
    q = np.array([[0.5], [1.5]], dtype=np.float64)
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D arrays, float32
    x = np.array([[[2.0, 2.5], [3.0, 3.5]], [[4.0, 4.5], [5.0, 5.5]]], dtype=np.float32)
    q = np.array([[[1.0, 1.1], [1.2, 1.3]], [[1.4, 1.5], [1.6, 1.7]]], dtype=np.float32)
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large x values, float32
    x = np.array([10.0, 20.0, 50.0], dtype=np.float32)
    q = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large q values, float64
    x = np.array([2.5, 3.5], dtype=np.float64)
    q = np.array([10.0, 100.0], dtype=np.float64)
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.special.zeta"] = zeta_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.zeta' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.zeta'.")


check_valid('jax.scipy.special.zeta', generated_inputs['jax.scipy.special.zeta'], lib="jax", suffix=0)

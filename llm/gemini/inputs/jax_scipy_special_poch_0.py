
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def poch_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, positive float32
    z = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    m = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    list_of_inputs.append({"z": z, "m": m})

    # Input 2: 0D arrays (scalars), float32
    z = np.array(2.0, dtype=np.float32)
    m = np.array(3.0, dtype=np.float32)
    list_of_inputs.append({"z": z, "m": m})

    # Input 3: 2D arrays, float32
    z = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    m = np.array([[0.5, 0.5], [1.5, 1.5]], dtype=np.float32)
    list_of_inputs.append({"z": z, "m": m})

    # Input 4: float64 arrays
    z = np.array([5.2, 6.8], dtype=np.float64)
    m = np.array([-1.2, 0.4], dtype=np.float64)
    list_of_inputs.append({"z": z, "m": m})

    # Input 5: Broadcasting shapes (3, 1) and (1, 4)
    z = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    m = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    list_of_inputs.append({"z": z, "m": m})

    # Input 6: Large 3D arrays
    z = np.random.uniform(1.0, 5.0, size=(2, 2, 2)).astype(np.float32)
    m = np.random.uniform(0.5, 2.0, size=(2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"z": z, "m": m})

    # Input 7: Small positive values
    z = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    m = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    list_of_inputs.append({"z": z, "m": m})

    # Input 8: Negative m values (where z + m > 0 to avoid gamma poles)
    z = np.array([5.0, 10.0], dtype=np.float32)
    m = np.array([-2.0, -4.5], dtype=np.float32)
    list_of_inputs.append({"z": z, "m": m})

    # Input 9: Large z and small m
    z = np.array([100.0, 200.0], dtype=np.float32)
    m = np.array([0.1, 0.2], dtype=np.float32)
    list_of_inputs.append({"z": z, "m": m})

    # Input 10: 1D array with positive float64 values
    z = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    m = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    list_of_inputs.append({"z": z, "m": m})

    return list_of_inputs

generated_inputs["jax.scipy.special.poch"] = poch_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.poch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.poch'.")


check_valid('jax.scipy.special.poch', generated_inputs['jax.scipy.special.poch'], lib="jax", suffix=0)

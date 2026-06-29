
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bessel_jn_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 array, order 3, default n_iter approximation
    z = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    v = 3
    n_iter = 50
    list_of_inputs.append({"z": z, "v": v, "n_iter": n_iter})

    # Input 2: 2D float32 array, order 1, smaller n_iter
    z = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    v = 1
    n_iter = 30
    list_of_inputs.append({"z": z, "v": v, "n_iter": n_iter})

    # Input 3: 1D array with negative float values (float64), order 2, n_iter=40
    z = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
    v = 2
    n_iter = 40
    list_of_inputs.append({"z": z, "v": v, "n_iter": n_iter})

    # Input 4: Scalar-like (0D) float32 array, order 0
    z = np.array(1.23, dtype=np.float32)
    v = 0
    n_iter = 20
    list_of_inputs.append({"z": z, "v": v, "n_iter": n_iter})

    # Input 5: 3D float32 array, order 4, larger n_iter
    z = np.random.uniform(0.1, 5.0, size=(2, 2, 2)).astype(np.float32)
    v = 4
    n_iter = 60
    list_of_inputs.append({"z": z, "v": v, "n_iter": n_iter})

    # Input 6: 1D float64 array with very small values, order 5, n_iter=80
    z = np.array([1e-5, 1e-4, 1e-3], dtype=np.float64)
    v = 5
    n_iter = 80
    list_of_inputs.append({"z": z, "v": v, "n_iter": n_iter})

    # Input 7: 2D float64 array, order 2, default-like n_iter
    z = np.random.uniform(-10.0, 10.0, size=(3, 3)).astype(np.float64)
    v = 2
    n_iter = 50
    list_of_inputs.append({"z": z, "v": v, "n_iter": n_iter})

    # Input 8: 1D float32 array with zero, order 1, low n_iter
    z = np.array([0.0], dtype=np.float32)
    v = 1
    n_iter = 10
    list_of_inputs.append({"z": z, "v": v, "n_iter": n_iter})

    # Input 9: 4D float32 array, order 3, n_iter=70
    z = np.random.uniform(1.0, 10.0, size=(2, 1, 2, 2)).astype(np.float32)
    v = 3
    n_iter = 70
    list_of_inputs.append({"z": z, "v": v, "n_iter": n_iter})

    # Input 10: 1D float32 array with larger values, order 2, n_iter=100
    z = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    v = 2
    n_iter = 100
    list_of_inputs.append({"z": z, "v": v, "n_iter": n_iter})

    return list_of_inputs

generated_inputs["jax.scipy.special.bessel_jn"] = bessel_jn_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.bessel_jn' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.bessel_jn'.")


check_valid('jax.scipy.special.bessel_jn', generated_inputs['jax.scipy.special.bessel_jn'], lib="jax", suffix=0)

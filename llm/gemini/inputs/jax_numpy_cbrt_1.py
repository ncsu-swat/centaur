
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def cbrt_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive, negative, and zero values
    x = np.array([-8.0, -1.0, 0.0, 1.0, 8.0, 27.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float64 array with positive values
    x = np.random.uniform(1.0, 1000.0, (3, 4)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D int32 array (JAX/NumPy converts int to float during cbrt)
    x = np.array([[[1, -1], [27, -27]], [[64, -64], [125, -125]]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 1D int16 array
    x = np.array([64, 125, 216, 343, 512], dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 0D (scalar-like) float32 array
    x = np.array(-27.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: Large 2D float32 array with random values
    x = np.random.uniform(-10000.0, 10000.0, (50, 50)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 4D float32 array
    x = np.random.uniform(-10.0, 10.0, (2, 3, 4, 5)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D float32 array containing special values (zeros, inf, nan)
    x = np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 2D int64 array
    x = np.random.randint(-1000, 1000, size=(5, 5)).astype(np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 5D float32 array with some dimensions of size 1
    x = np.random.uniform(1.0, 100.0, (1, 2, 1, 3, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.cbrt_1"] = cbrt_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cbrt_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cbrt_1'.")


check_valid('jax.numpy.cbrt', generated_inputs['jax.numpy.cbrt_1'], lib="jax", suffix=1)

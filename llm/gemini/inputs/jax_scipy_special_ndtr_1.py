
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ndtr_inputs():
    list_of_inputs = []

    # Input 1: 0D array (scalar) float32, positive value
    x1 = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": x1})

    # Input 2: 0D array (scalar) float64, negative value
    x2 = np.array(-2.5, dtype=np.float64)
    list_of_inputs.append({"x": x2})

    # Input 3: 1D array float32 with positive, negative and zero
    x3 = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"x": x3})

    # Input 4: 1D array float64 with extreme values
    x4 = np.array([-10.0, -5.0, 5.0, 10.0], dtype=np.float64)
    list_of_inputs.append({"x": x4})

    # Input 5: 2D array float32 with random standard normal values
    x5 = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({"x": x5})

    # Input 6: 2D array float64 with random values between -1 and 1
    x6 = np.random.uniform(-1.0, 1.0, (2, 5)).astype(np.float64)
    list_of_inputs.append({"x": x6})

    # Input 7: 3D array float32, small size
    x7 = np.random.randn(2, 2, 3).astype(np.float32)
    list_of_inputs.append({"x": x7})

    # Input 8: 4D array float64
    x8 = np.random.randn(2, 2, 2, 2).astype(np.float64)
    list_of_inputs.append({"x": x8})

    # Input 9: 1D array float32 with values very close to zero
    x9 = np.array([-1e-5, 0.0, 1e-5], dtype=np.float32)
    list_of_inputs.append({"x": x9})

    # Input 10: 2D array float32 with large range of values
    x10 = np.linspace(-5.0, 5.0, 12).reshape(3, 4).astype(np.float32)
    list_of_inputs.append({"x": x10})

    return list_of_inputs

generated_inputs["jax.scipy.special.ndtr_1"] = ndtr_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.ndtr_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.ndtr_1'.")


check_valid('jax.scipy.special.ndtr', generated_inputs['jax.scipy.special.ndtr_1'], lib="jax", suffix=1)

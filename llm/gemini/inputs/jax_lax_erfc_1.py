
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_erfc_inputs():
    list_of_inputs = []

    # Input 1: Standard 1D float32 array
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 2: 2D float32 array with positive values
    x = np.random.uniform(0.1, 5.0, size=(3, 3)).astype(np.float32)
    list_of_inputs.append({"x": x})

    # Input 3: 3D float64 array with both positive and negative values
    x = np.random.normal(0.0, 2.0, size=(2, 4, 3)).astype(np.float64)
    list_of_inputs.append({"x": x})

    # Input 4: 1D float16 array
    x = np.array([-0.5, 0.5, 1.5, -1.5], dtype=np.float16)
    list_of_inputs.append({"x": x})

    # Input 5: Scalar (0D array)
    x = np.array(0.75, dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 6: 4D float32 array with extreme values
    x = np.array([[[[-10.0, -5.0], [0.0, 5.0]], [[10.0, 20.0], [-20.0, -10.0]]]]).astype(np.float32)
    list_of_inputs.append({"x": x})

    # Input 7: 2D float32 array containing very small values close to zero
    x = np.random.uniform(-1e-5, 1e-5, size=(5, 5)).astype(np.float32)
    list_of_inputs.append({"x": x})

    # Input 8: 1D float32 array
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 9: 5D float32 array of ones multiplied by a scale factor
    x = np.ones((1, 2, 2, 2, 1), dtype=np.float32) * 1.5
    list_of_inputs.append({"x": x})

    # Input 10: 1D float32 array with infinity and NaN values to check edge cases
    x = np.array([-np.inf, np.nan, np.inf], dtype=np.float32)
    list_of_inputs.append({"x": x})

    return list_of_inputs

generated_inputs["jax.lax.erfc_1"] = jax_lax_erfc_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.erfc_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.erfc_1'.")


check_valid('jax.lax.erfc', generated_inputs['jax.lax.erfc_1'], lib="jax", suffix=1)

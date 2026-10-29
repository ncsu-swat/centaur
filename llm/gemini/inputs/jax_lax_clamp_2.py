
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def clamp_inputs():
    list_of_inputs = []

    # Input 1: 1D array, standard range
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
    min_val = -1.0
    max_val = 1.0
    list_of_inputs.append(
        copy.deepcopy({"min": min_val, "x": x, "max": max_val})
    )

    # Input 2: 2D array, positive clamp range
    x = np.random.randn(3, 4).astype(np.float32) * 5.0
    min_val = 0.0
    max_val = 2.5
    list_of_inputs.append(
        copy.deepcopy({"min": min_val, "x": x, "max": max_val})
    )

    # Input 3: 0D scalar array
    x = np.array(3.1415, dtype=np.float32)
    min_val = -1.0
    max_val = 2.0
    list_of_inputs.append(
        copy.deepcopy({"min": min_val, "x": x, "max": max_val})
    )

    # Input 4: 3D array, float64
    x = np.random.randn(2, 3, 4).astype(np.float64) * 10.0
    min_val = -5.0
    max_val = 5.0
    list_of_inputs.append(
        copy.deepcopy({"min": min_val, "x": x, "max": max_val})
    )

    # Input 5: 4D array, narrow range
    x = np.random.uniform(-10.0, 10.0, size=(2, 2, 3, 3)).astype(np.float32)
    min_val = -0.5
    max_val = 0.5
    list_of_inputs.append(
        copy.deepcopy({"min": min_val, "x": x, "max": max_val})
    )

    # Input 6: 1D array with negative range
    x = np.linspace(-10.0, 0.0, 20, dtype=np.float32)
    min_val = -8.0
    max_val = -2.0
    list_of_inputs.append(
        copy.deepcopy({"min": min_val, "x": x, "max": max_val})
    )

    # Input 7: 2D array, min equals max
    x = np.random.randn(4, 4).astype(np.float32)
    min_val = 1.5
    max_val = 1.5
    list_of_inputs.append(
        copy.deepcopy({"min": min_val, "x": x, "max": max_val})
    )

    # Input 8: 1D array containing infinities
    x = np.array([-np.inf, -100.0, 0.0, 100.0, np.inf], dtype=np.float32)
    min_val = -10.0
    max_val = 10.0
    list_of_inputs.append(
        copy.deepcopy({"min": min_val, "x": x, "max": max_val})
    )

    # Input 9: 5D array, float64
    x = np.random.randn(1, 2, 1, 3, 2).astype(np.float64)
    min_val = -2.0
    max_val = 2.0
    list_of_inputs.append(
        copy.deepcopy({"min": min_val, "x": x, "max": max_val})
    )

    # Input 10: 2D array, large range
    x = np.array([[-1000.0, 500.0], [200.0, -300.0]], dtype=np.float32)
    min_val = -250.0
    max_val = 250.0
    list_of_inputs.append(
        copy.deepcopy({"min": min_val, "x": x, "max": max_val})
    )

    return list_of_inputs


generated_inputs["jax.lax.clamp_2"] = clamp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.clamp_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.clamp_2'.")


check_valid('jax.lax.clamp', generated_inputs['jax.lax.clamp_2'], lib="jax", suffix=2)

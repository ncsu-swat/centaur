
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gelu_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, approximate=True
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    approximate = True
    list_of_inputs.append(
        copy.deepcopy({"x": x, "approximate": approximate})
    )

    # Input 2: 1D array, float32, approximate=False
    x = np.array([-2.5, -0.5, 0.0, 0.5, 2.5], dtype=np.float32)
    approximate = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "approximate": approximate})
    )

    # Input 3: 2D array, float64, approximate=True
    x = np.random.randn(4, 5).astype(np.float64)
    approximate = True
    list_of_inputs.append(
        copy.deepcopy({"x": x, "approximate": approximate})
    )

    # Input 4: 2D array, float64, approximate=False
    x = np.random.randn(3, 3).astype(np.float64)
    approximate = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "approximate": approximate})
    )

    # Input 5: 3D array, float32, approximate=True
    x = np.random.uniform(-5.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    approximate = True
    list_of_inputs.append(
        copy.deepcopy({"x": x, "approximate": approximate})
    )

    # Input 6: 3D array, float16, approximate=False
    x = np.random.randn(2, 2, 2).astype(np.float16)
    approximate = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "approximate": approximate})
    )

    # Input 7: 4D array, float32, approximate=True
    x = np.random.randn(2, 4, 8, 8).astype(np.float32)
    approximate = True
    list_of_inputs.append(
        copy.deepcopy({"x": x, "approximate": approximate})
    )

    # Input 8: 0D scalar tensor, float32, approximate=False
    x = np.array(1.5, dtype=np.float32)
    approximate = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "approximate": approximate})
    )

    # Input 9: Large negative values, float32, approximate=True
    x = np.array([-100.0, -50.0, -10.0, -5.0], dtype=np.float32)
    approximate = True
    list_of_inputs.append(
        copy.deepcopy({"x": x, "approximate": approximate})
    )

    # Input 10: Large positive values, float32, approximate=False
    x = np.array([5.0, 10.0, 50.0, 100.0], dtype=np.float32)
    approximate = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "approximate": approximate})
    )

    # Input 11: Zeros array, float32, approximate=True
    x = np.zeros((3, 3), dtype=np.float32)
    approximate = True
    list_of_inputs.append(
        copy.deepcopy({"x": x, "approximate": approximate})
    )

    return list_of_inputs


generated_inputs["jax.nn.gelu"] = gelu_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.gelu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.gelu'.")


check_valid('jax.nn.gelu', generated_inputs['jax.nn.gelu'], lib="jax", suffix=0)

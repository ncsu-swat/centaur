
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class EmptyAccuracy(tuple):

    def __new__(cls):
        return super().__new__(cls, ())

    def __bool__(self):
        return False

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    def __hash__(self):
        return 0


def exp_inputs():
    list_of_inputs = []

    # Input 1: 0D float32 scalar
    x = np.array(1.5, dtype=np.float32)
    acc = EmptyAccuracy()
    list_of_inputs.append({"x": copy.deepcopy(x), "accuracy": acc})

    # Input 2: 1D float32 array with mixed positive and negative values
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    acc = EmptyAccuracy()
    list_of_inputs.append({"x": copy.deepcopy(x), "accuracy": acc})

    # Input 3: 2D float64 array
    x = np.array([[-1.5, 0.5], [2.0, -0.5]], dtype=np.float64)
    acc = EmptyAccuracy()
    list_of_inputs.append({"x": copy.deepcopy(x), "accuracy": acc})

    # Input 4: 1D float32 array strictly negative
    x = np.array([-10.0, -5.0, -2.0, -0.5], dtype=np.float32)
    acc = EmptyAccuracy()
    list_of_inputs.append({"x": copy.deepcopy(x), "accuracy": acc})

    # Input 5: 3D float32 array
    x = np.random.uniform(-3.0, 3.0, size=(2, 3, 4)).astype(np.float32)
    acc = EmptyAccuracy()
    list_of_inputs.append({"x": copy.deepcopy(x), "accuracy": acc})

    # Input 6: 2D float64 array of zeros
    x = np.zeros((3, 3), dtype=np.float64)
    acc = EmptyAccuracy()
    list_of_inputs.append({"x": copy.deepcopy(x), "accuracy": acc})

    # Input 7: 1D float64 array with small values
    x = np.array([-0.01, -0.001, 0.0, 0.001, 0.01], dtype=np.float64)
    acc = EmptyAccuracy()
    list_of_inputs.append({"x": copy.deepcopy(x), "accuracy": acc})

    # Input 8: 4D float32 array
    x = np.random.uniform(-1.0, 1.0, size=(1, 2, 2, 3)).astype(np.float32)
    acc = EmptyAccuracy()
    list_of_inputs.append({"x": copy.deepcopy(x), "accuracy": acc})

    # Input 9: 2D float32 array
    x = np.array([[-3.0, -2.0], [-1.0, 0.0]], dtype=np.float32)
    acc = EmptyAccuracy()
    list_of_inputs.append({"x": copy.deepcopy(x), "accuracy": acc})

    # Input 10: 3D float64 array
    x = np.random.randn(2, 2, 2).astype(np.float64)
    acc = EmptyAccuracy()
    list_of_inputs.append({"x": copy.deepcopy(x), "accuracy": acc})

    # Input 11: 1D float32 array with moderately large negative values
    x = np.array([-20.0, -10.0, -5.0], dtype=np.float32)
    acc = EmptyAccuracy()
    list_of_inputs.append({"x": copy.deepcopy(x), "accuracy": acc})

    return list_of_inputs


generated_inputs["jax.lax.exp_2"] = exp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.exp_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.exp_2'.")


check_valid('jax.lax.exp', generated_inputs['jax.lax.exp_2'], lib="jax", suffix=2)

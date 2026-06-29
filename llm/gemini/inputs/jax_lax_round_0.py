
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class MagicStr(str):
    def __new__(cls, string_val, int_val):
        return str.__new__(cls, string_val)
    def __init__(self, string_val, int_val):
        self.int_val = int_val
    def __hash__(self):
        return hash(self.int_val)
    def __eq__(self, other):
        if isinstance(other, int):
            return self.int_val == other
        return super().__eq__(other)

def jax_lax_round_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, AWAY_FROM_ZERO
    x = np.array([-1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float32)
    rounding_method = MagicStr("AWAY_FROM_ZERO", 0)
    list_of_inputs.append({"x": copy.deepcopy(x), "rounding_method": rounding_method})

    # Input 2: 1D float32 array, TO_NEAREST_EVEN
    x = np.array([-1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float32)
    rounding_method = MagicStr("TO_NEAREST_EVEN", 1)
    list_of_inputs.append({"x": copy.deepcopy(x), "rounding_method": rounding_method})

    # Input 3: 2D float64 array, AWAY_FROM_ZERO
    x = np.array([[-2.7, -1.2, 0.0], [1.2, 2.7, 3.5]], dtype=np.float64)
    rounding_method = MagicStr("AWAY_FROM_ZERO", 0)
    list_of_inputs.append({"x": copy.deepcopy(x), "rounding_method": rounding_method})

    # Input 4: 2D float32 array, TO_NEAREST_EVEN
    x = np.array([[-2.5, -1.5], [0.5, 1.5]], dtype=np.float32)
    rounding_method = MagicStr("TO_NEAREST_EVEN", 1)
    list_of_inputs.append({"x": copy.deepcopy(x), "rounding_method": rounding_method})

    # Input 5: 3D float16 array, AWAY_FROM_ZERO
    x = np.array([[[-0.1, 0.1], [0.9, -0.9]], [[1.5, -1.5], [2.1, -2.1]]], dtype=np.float16)
    rounding_method = MagicStr("AWAY_FROM_ZERO", 0)
    list_of_inputs.append({"x": copy.deepcopy(x), "rounding_method": rounding_method})

    # Input 6: 4D float32 array, TO_NEAREST_EVEN
    x = np.random.uniform(-10.0, 10.0, size=(2, 2, 3, 3)).astype(np.float32)
    rounding_method = MagicStr("TO_NEAREST_EVEN", 1)
    list_of_inputs.append({"x": copy.deepcopy(x), "rounding_method": rounding_method})

    # Input 7: 1D float64 array, AWAY_FROM_ZERO
    x = np.array([0.4999, 0.5001, -0.4999, -0.5001], dtype=np.float64)
    rounding_method = MagicStr("AWAY_FROM_ZERO", 0)
    list_of_inputs.append({"x": copy.deepcopy(x), "rounding_method": rounding_method})

    # Input 8: Scalar-like 1D float32 array, TO_NEAREST_EVEN
    x = np.array([3.5], dtype=np.float32)
    rounding_method = MagicStr("TO_NEAREST_EVEN", 1)
    list_of_inputs.append({"x": copy.deepcopy(x), "rounding_method": rounding_method})

    # Input 9: Large values in 2D float32 array, AWAY_FROM_ZERO
    x = np.array([[1000.5, -1000.5], [10001.5, -10001.5]], dtype=np.float32)
    rounding_method = MagicStr("AWAY_FROM_ZERO", 0)
    list_of_inputs.append({"x": copy.deepcopy(x), "rounding_method": rounding_method})

    # Input 10: 3D float32 array, TO_NEAREST_EVEN
    x = np.random.uniform(-5.0, 5.0, size=(1, 5, 5)).astype(np.float32)
    rounding_method = MagicStr("TO_NEAREST_EVEN", 1)
    list_of_inputs.append({"x": copy.deepcopy(x), "rounding_method": rounding_method})

    return list_of_inputs

generated_inputs["jax.lax.round"] = jax_lax_round_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.round' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.round'.")


check_valid('jax.lax.round', generated_inputs['jax.lax.round'], lib="jax", suffix=0)

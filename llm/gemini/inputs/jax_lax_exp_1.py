
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
import numpy as np


class AccuracyStr(str):

    def __new__(cls, val="DEFAULT"):
        return super().__new__(cls, val)

    def __eq__(self, other):
        if hasattr(jax.lax, "AccuracyMode") and hasattr(
            jax.lax.AccuracyMode, "DEFAULT"
        ):
            if (
                other is jax.lax.AccuracyMode.DEFAULT
                or other == jax.lax.AccuracyMode.DEFAULT
            ):
                return True
        return str(other) in ("DEFAULT", "default")

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash("DEFAULT")

    def __bool__(self):
        return False


def exp_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    x = np.array([-2.0, -0.5, 0.0, 1.0, 2.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "accuracy": AccuracyStr("DEFAULT")})

    # Input 2: 0D scalar float32
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": x, "accuracy": AccuracyStr("DEFAULT")})

    # Input 3: 2D float64 array
    x = np.array([[-1.0, 2.0], [0.5, -0.5]], dtype=np.float64)
    list_of_inputs.append({"x": x, "accuracy": AccuracyStr("DEFAULT")})

    # Input 4: 1D float16 array
    x = np.array([-3.0, -1.0, 0.0, 2.0], dtype=np.float16)
    list_of_inputs.append({"x": x, "accuracy": AccuracyStr("DEFAULT")})

    # Input 5: 3D float32 array with negative and positive values
    x = np.random.uniform(-3.0, 3.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": x, "accuracy": AccuracyStr("DEFAULT")})

    # Input 6: 1D complex64 array
    x = np.array([1.0 + 2.0j, -0.5 - 1.5j, 0.0 + 0.0j], dtype=np.complex64)
    list_of_inputs.append({"x": x, "accuracy": AccuracyStr("DEFAULT")})

    # Input 7: 2D complex128 array
    x = np.array(
        [[1.0 - 1.0j, -2.0 + 0.5j], [0.0 + 3.0j, 1.5 + 0.0j]],
        dtype=np.complex128,
    )
    list_of_inputs.append({"x": x, "accuracy": AccuracyStr("DEFAULT")})

    # Input 8: 4D float32 array
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append({"x": x, "accuracy": AccuracyStr("DEFAULT")})

    # Input 9: 2D float32 array with zeros
    x = np.zeros((3, 5), dtype=np.float32)
    list_of_inputs.append({"x": x, "accuracy": AccuracyStr("DEFAULT")})

    # Input 10: 5D float32 array
    x = np.random.uniform(-2.0, 2.0, size=(1, 2, 2, 3, 2)).astype(np.float32)
    list_of_inputs.append({"x": x, "accuracy": AccuracyStr("DEFAULT")})

    # Input 11: 1D float64 array with larger range
    x = np.linspace(-10.0, 10.0, 15, dtype=np.float64)
    list_of_inputs.append({"x": x, "accuracy": AccuracyStr("DEFAULT")})

    return list_of_inputs


generated_inputs["jax.lax.exp_1"] = exp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.exp_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.exp_1'.")


check_valid('jax.lax.exp', generated_inputs['jax.lax.exp_1'], lib="jax", suffix=1)

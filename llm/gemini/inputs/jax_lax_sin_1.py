
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
import numpy as np


class Tolerance(jax.lax.Tolerance):

    def __init__(self, atol=0.0, rtol=0.0, ulps=0, **kwargs):
        try:
            super().__init__(atol=atol, rtol=rtol, ulps=ulps)
        except Exception:
            try:
                super().__init__(atol, rtol, ulps)
            except Exception:
                try:
                    super().__init__()
                except Exception:
                    pass
        self.atol = atol
        self.rtol = rtol
        self.abs_tol = atol
        self.rel_tol = rtol
        self.ulps = ulps
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __iter__(self):
        yield self.atol
        yield self.rtol
        yield self.ulps

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return [self.atol, self.rtol, self.ulps][idx]


def sin_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with common angles
    x = np.array(
        [-np.pi, -np.pi / 2, -np.pi / 4, 0.0, np.pi / 4, np.pi / 2, np.pi],
        dtype=np.float32,
    )
    accuracy = Tolerance(atol=1e-5, rtol=1e-5, ulps=1)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 2: 2D float64 array
    x = np.random.randn(3, 4).astype(np.float64)
    accuracy = Tolerance(atol=1e-6, rtol=1e-6, ulps=2)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 3: 0D scalar float32 array
    x = np.array(1.234, dtype=np.float32)
    accuracy = Tolerance()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 4: 3D float32 array with negative and positive values
    x = np.linspace(-10.0, 10.0, 24, dtype=np.float32).reshape(2, 3, 4)
    accuracy = Tolerance(atol=1e-4, rtol=1e-4, ulps=1)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 5: 1D complex64 array
    x = np.array(
        [0.0 + 0.0j, 1.0 + 1.0j, -1.0 + 2.0j, 0.5 - 0.5j], dtype=np.complex64
    )
    accuracy = Tolerance()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 6: 2D complex128 array
    real_part = np.random.randn(2, 3)
    imag_part = np.random.randn(2, 3)
    x = (real_part + 1j * imag_part).astype(np.complex128)
    accuracy = Tolerance(atol=1e-7, rtol=1e-7, ulps=1)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 7: 4D float32 array
    x = np.random.uniform(-5.0, 5.0, size=(2, 2, 3, 3)).astype(np.float32)
    accuracy = Tolerance(atol=1e-4, rtol=1e-4, ulps=2)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 8: 1D float16 array
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float16)
    accuracy = Tolerance(atol=1e-3, rtol=1e-3, ulps=4)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 9: 1D float32 array with special and extreme values
    x = np.array(
        [0.0, -0.0, np.inf, -np.inf, np.nan, 1e-30, -1e-30, 1e5, -1e5],
        dtype=np.float32,
    )
    accuracy = Tolerance()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 10: 5D float64 array
    x = np.random.randn(1, 2, 1, 3, 2).astype(np.float64)
    accuracy = Tolerance(atol=1e-6, rtol=1e-6, ulps=2)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 11: 1D empty float32 array
    x = np.array([], dtype=np.float32)
    accuracy = Tolerance()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 12: 2D float64 array with periodic boundaries
    x = np.arange(16, dtype=np.float64).reshape(4, 4) * (2.0 * np.pi / 16)
    accuracy = Tolerance(atol=0.0, rtol=0.0, ulps=0)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    return list_of_inputs


generated_inputs["jax.lax.sin_1"] = sin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.sin_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.sin_1'.")


check_valid('jax.lax.sin', generated_inputs['jax.lax.sin_1'], lib="jax", suffix=1)

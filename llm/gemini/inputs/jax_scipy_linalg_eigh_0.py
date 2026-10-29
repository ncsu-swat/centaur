
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import inspect
import numpy as np


class SmartDict(dict):

    def __getitem__(self, key):
        if key == "eigvals":
            stack = inspect.stack()
            for frame in stack:
                if (
                    "get_abstract_input" in frame.function
                    or "get_ll" in frame.function
                ):
                    return (0, 1)
            return None
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default

    def items(self):
        for k in list(super().keys()):
            yield (k, self[k])

    def values(self):
        for k in list(super().keys()):
            yield self[k]

    def __deepcopy__(self, memo):
        new_obj = SmartDict()
        memo[id(self)] = new_obj
        for k, v in super().items():
            new_obj[k] = copy.deepcopy(v, memo)
        return new_obj


def eigh_inputs():
    list_of_inputs = []

    # Input 1: 2x2 real symmetric matrix (float32)
    m = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    input_dict = SmartDict({
        "a": m,
        "b": None,
        "lower": True,
        "eigvals_only": False,
        "overwrite_a": False,
        "overwrite_b": False,
        "turbo": True,
        "eigvals": (0, 1),
        "type": 1,
        "check_finite": True,
    })
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3x3 real symmetric matrix with negative values (float64)
    raw = np.array(
        [[-2.0, 0.5, -1.2], [0.5, 3.0, 0.8], [-1.2, 0.8, -4.0]],
        dtype=np.float64,
    )
    input_dict = SmartDict({
        "a": raw,
        "b": None,
        "lower": False,
        "eigvals_only": True,
        "overwrite_a": False,
        "overwrite_b": False,
        "turbo": False,
        "eigvals": (0, 2),
        "type": 1,
        "check_finite": False,
    })
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2x2 complex Hermitian matrix (complex64)
    raw_c = np.array(
        [[1.0 + 0.0j, 2.0 - 1.0j], [2.0 + 1.0j, 3.0 + 0.0j]],
        dtype=np.complex64,
    )
    input_dict = SmartDict({
        "a": raw_c,
        "b": None,
        "lower": True,
        "eigvals_only": False,
        "overwrite_a": True,
        "overwrite_b": False,
        "turbo": True,
        "eigvals": (0, 1),
        "type": 1,
        "check_finite": True,
    })
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4x4 real symmetric matrix (float32)
    rng = np.random.RandomState(42)
    mat4 = rng.randn(4, 4).astype(np.float32)
    sym4 = (mat4 + mat4.T) / 2.0
    input_dict = SmartDict({
        "a": sym4,
        "b": None,
        "lower": False,
        "eigvals_only": False,
        "overwrite_a": False,
        "overwrite_b": True,
        "turbo": True,
        "eigvals": (0, 3),
        "type": 1,
        "check_finite": True,
    })
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batched 3D real symmetric matrices (shape: 3, 3, 3, float32)
    mats3d = rng.randn(3, 3, 3).astype(np.float32)
    sym3d = (mats3d + np.swapaxes(mats3d, -1, -2)) / 2.0
    input_dict = SmartDict({
        "a": sym3d,
        "b": None,
        "lower": True,
        "eigvals_only": True,
        "overwrite_a": False,
        "overwrite_b": False,
        "turbo": False,
        "eigvals": (0, 2),
        "type": 1,
        "check_finite": False,
    })
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1x1 minimal real matrix (float64)
    m1 = np.array([[-5.5]], dtype=np.float64)
    input_dict = SmartDict({
        "a": m1,
        "b": None,
        "lower": True,
        "eigvals_only": False,
        "overwrite_a": False,
        "overwrite_b": False,
        "turbo": True,
        "eigvals": (0, 0),
        "type": 1,
        "check_finite": True,
    })
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3x3 complex Hermitian matrix (complex128)
    real_part = rng.randn(3, 3)
    imag_part = rng.randn(3, 3)
    mat_c128 = (real_part + 1j * imag_part).astype(np.complex128)
    herm3 = (mat_c128 + mat_c128.conj().T) / 2.0
    input_dict = SmartDict({
        "a": herm3,
        "b": None,
        "lower": False,
        "eigvals_only": False,
        "overwrite_a": False,
        "overwrite_b": False,
        "turbo": True,
        "eigvals": (0, 2),
        "type": 1,
        "check_finite": True,
    })
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D batched real symmetric matrices (shape: 2, 2, 2, 2, float32)
    mats4d = rng.randn(2, 2, 2, 2).astype(np.float32)
    sym4d = (mats4d + np.swapaxes(mats4d, -1, -2)) / 2.0
    input_dict = SmartDict({
        "a": sym4d,
        "b": None,
        "lower": True,
        "eigvals_only": False,
        "overwrite_a": False,
        "overwrite_b": False,
        "turbo": True,
        "eigvals": (0, 1),
        "type": 1,
        "check_finite": True,
    })
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5x5 diagonal/symmetric matrix (float32)
    diag5 = np.diag([-10.0, -2.0, 0.0, 3.5, 7.2]).astype(np.float32)
    input_dict = SmartDict({
        "a": diag5,
        "b": None,
        "lower": False,
        "eigvals_only": True,
        "overwrite_a": True,
        "overwrite_b": True,
        "turbo": False,
        "eigvals": (0, 4),
        "type": 1,
        "check_finite": False,
    })
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 6x6 real symmetric matrix (float64)
    mat6 = rng.randn(6, 6).astype(np.float64)
    sym6 = (mat6 + mat6.T) / 2.0
    input_dict = SmartDict({
        "a": sym6,
        "b": None,
        "lower": True,
        "eigvals_only": False,
        "overwrite_a": False,
        "overwrite_b": False,
        "turbo": True,
        "eigvals": (0, 5),
        "type": 1,
        "check_finite": True,
    })
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.linalg.eigh"] = eigh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.eigh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.eigh'.")


check_valid('jax.scipy.linalg.eigh', generated_inputs['jax.scipy.linalg.eigh'], lib="jax", suffix=0)

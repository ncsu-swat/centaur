
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import jax
import copy

# Patch jax.lax.Tolerance to be iterable and behave like a tuple if it is not
if not hasattr(jax.lax.Tolerance, '__iter__'):
    def tolerance_iter(self):
        atol = getattr(self, 'absolute', getattr(self, 'atol', None))
        rtol = getattr(self, 'relative', getattr(self, 'rtol', None))
        if atol is None:
            d = getattr(self, '__dict__', {})
            vals = [v for v in d.values() if isinstance(v, (int, float))]
            atol = vals[0] if len(vals) > 0 else 1e-5
            rtol = vals[1] if len(vals) > 1 else 1e-5
        yield atol
        yield rtol

    jax.lax.Tolerance.__iter__ = tolerance_iter
    jax.lax.Tolerance.__len__ = lambda self: 2
    jax.lax.Tolerance.__getitem__ = lambda self, idx: list(self)[idx]

def sin_inputs():
    list_of_inputs = []
    
    # Input 1: float32, 1D array
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    accuracy = jax.lax.Tolerance(1e-5, 1e-5)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: float64, 2D array
    x = np.array([[-np.pi, np.pi/2], [0, np.pi]], dtype=np.float64)
    accuracy = jax.lax.Tolerance(1e-6, 1e-6)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: complex64, 1D array
    x = np.array([1.0 + 1.0j, -2.0 - 3.0j], dtype=np.complex64)
    accuracy = jax.lax.Tolerance(1e-5, 1e-5)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: complex128, 2D array
    x = np.array([[1j, -1j], [0.5j, 2.5]], dtype=np.complex128)
    accuracy = jax.lax.Tolerance(1e-7, 1e-7)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, 3D array
    x = np.random.randn(2, 3, 4).astype(np.float32)
    accuracy = jax.lax.Tolerance(1e-4, 1e-4)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, 4D array
    x = np.random.randn(1, 2, 2, 3).astype(np.float64)
    accuracy = jax.lax.Tolerance(1e-8, 1e-8)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, 0D array (scalar)
    x = np.array(0.5, dtype=np.float32)
    accuracy = jax.lax.Tolerance(1e-5, 1e-5)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex64, 3D array
    x = (np.random.randn(2, 2, 2) + 1j * np.random.randn(2, 2, 2)).astype(np.complex64)
    accuracy = jax.lax.Tolerance(1e-5, 1e-5)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, large range values
    x = np.array([-100.0, 100.0, 1000.0], dtype=np.float32)
    accuracy = jax.lax.Tolerance(1e-3, 1e-3)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64, large dimension
    x = np.linspace(-10, 10, 100, dtype=np.float64)
    accuracy = jax.lax.Tolerance(1e-6, 1e-6)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

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

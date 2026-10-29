
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax.numpy as jnp
import numpy as np


class CallableTensor:

    def __init__(self, func):
        self.func = func
        self.shape = ()
        self.dtype = np.dtype(np.float32)
        self.ndim = 0

    def __call__(self, x):
        return self.func(x)

    def __deepcopy__(self, memo):
        return CallableTensor(self.func)


def funm_inputs():
    list_of_inputs = []

    # Input 1: 2x2 float32 with jnp.cos, disp=True
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    func = CallableTensor(jnp.cos)
    list_of_inputs.append(copy.deepcopy({"A": A, "func": func, "disp": True}))

    # Input 2: 2x2 float32 with jnp.exp, disp=False
    A = np.array([[0.5, -1.0], [1.5, 2.0]], dtype=np.float32)
    func = CallableTensor(jnp.exp)
    list_of_inputs.append(copy.deepcopy({"A": A, "func": func, "disp": False}))

    # Input 3: 1x1 float32 scalar matrix with jnp.sin, disp=True
    A = np.array([[2.5]], dtype=np.float32)
    func = CallableTensor(jnp.sin)
    list_of_inputs.append(copy.deepcopy({"A": A, "func": func, "disp": True}))

    # Input 4: 3x3 float64 with jnp.sinh, disp=True
    A = np.array(
        [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]],
        dtype=np.float64,
    )
    func = CallableTensor(jnp.sinh)
    list_of_inputs.append(copy.deepcopy({"A": A, "func": func, "disp": True}))

    # Input 5: 3x3 float32 with jnp.cosh, disp=False
    A = np.array(
        [[1.0, 0.0, 2.0], [0.0, 3.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float32
    )
    func = CallableTensor(jnp.cosh)
    list_of_inputs.append(copy.deepcopy({"A": A, "func": func, "disp": False}))

    # Input 6: 4x4 float32 random matrix with jnp.exp, disp=True
    np.random.seed(42)
    A = np.random.randn(4, 4).astype(np.float32)
    func = CallableTensor(jnp.exp)
    list_of_inputs.append(copy.deepcopy({"A": A, "func": func, "disp": True}))

    # Input 7: 2x2 complex64 with jnp.exp, disp=True
    A = np.array(
        [[1.0 + 2.0j, -1.0j], [2.0j, 3.0 - 1.0j]], dtype=np.complex64
    )
    func = CallableTensor(jnp.exp)
    list_of_inputs.append(copy.deepcopy({"A": A, "func": func, "disp": True}))

    # Input 8: 2x2 float64 with custom linear combination, disp=False
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    func = CallableTensor(lambda x: jnp.sin(x) + 2.0 * jnp.cos(x))
    list_of_inputs.append(copy.deepcopy({"A": A, "func": func, "disp": False}))

    # Input 9: 3x3 complex128 with jnp.sin, disp=True
    A = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(
        np.complex128
    )
    func = CallableTensor(jnp.sin)
    list_of_inputs.append(copy.deepcopy({"A": A, "func": func, "disp": True}))

    # Input 10: 5x5 float32 symmetric matrix with square function, disp=True
    base = np.random.randn(5, 5).astype(np.float32)
    A = base + base.T
    func = CallableTensor(lambda x: x**2)
    list_of_inputs.append(copy.deepcopy({"A": A, "func": func, "disp": True}))

    # Input 11: 4x4 float64 diagonal matrix with jnp.cos, disp=False
    A = np.diag([1.0, -2.0, 3.0, -4.0]).astype(np.float64)
    func = CallableTensor(jnp.cos)
    list_of_inputs.append(copy.deepcopy({"A": A, "func": func, "disp": False}))

    return list_of_inputs


generated_inputs["jax.scipy.linalg.funm"] = funm_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.funm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.funm'.")


check_valid('jax.scipy.linalg.funm', generated_inputs['jax.scipy.linalg.funm'], lib="jax", suffix=0)

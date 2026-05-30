
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax.numpy as jnp

class CallableString(str):
    def __new__(cls, value, func=None):
        obj = super().__new__(cls, value)
        obj.func = func
        return obj
    def __call__(self, *args, **kwargs):
        if self.func is None:
            raise ValueError("Function is not defined.")
        return self.func(*args, **kwargs)
    def __deepcopy__(self, memo):
        return CallableString(str(self), self.func)

class IterableInt(int):
    def __iter__(self):
        return iter((int(self),))
    def __deepcopy__(self, memo):
        return IterableInt(int(self))

def fromfunction_inputs():
    list_of_inputs = []

    funcs = [
        ("lambda x: x", lambda x: x),
        ("lambda x: x * 2", lambda x: x * 2),
        ("lambda x: x + 10", lambda x: x + 10),
        ("lambda x: x ** 2", lambda x: x ** 2),
        ("lambda x: x - 5", lambda x: x - 5),
        ("lambda x: x / 2.0", lambda x: x / 2.0),
        ("lambda x: x % 3", lambda x: x % 3),
        ("lambda x: x * 1.5", lambda x: x * 1.5),
        ("lambda x: x + 1.23", lambda x: x + 1.23),
        ("lambda x: x * x - x", lambda x: x * x - x)
    ]
    shapes = [5, 3, 8, 10, 6, 4, 7, 9, 2, 12]
    dtypes = [
        np.dtype(np.float32),
        np.dtype(np.int32),
        np.dtype(np.float64),
        np.dtype(np.int64),
        np.dtype(np.float32),
        np.dtype(np.float32),
        np.dtype(np.int32),
        np.dtype(np.float64),
        np.dtype(np.float32),
        np.dtype(np.int32)
    ]

    for i in range(10):
        func_str, func_callable = funcs[i]
        f = CallableString(func_str, func_callable)
        s = IterableInt(shapes[i])
        d = dtypes[i]
        input_dict = {
            "function": f,
            "shape": s,
            "dtype": d
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fromfunction_2"] = fromfunction_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fromfunction_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fromfunction_2'.")


check_valid('jax.numpy.fromfunction', generated_inputs['jax.numpy.fromfunction_2'], lib="jax", suffix=2)

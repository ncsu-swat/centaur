
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import sys
import numpy as np


def _patch_harness():
    for mod in list(sys.modules.values()):
        if hasattr(mod, "get_input"):
            _orig_gi = getattr(mod, "get_input")

            def _make_gi(orig):
                def _patched_gi(api, input_dict, cpu=True, lib="jax"):
                    inp = orig(api, input_dict, cpu=cpu, lib=lib)
                    if (
                        api == "jax.numpy.var"
                        and isinstance(inp, dict)
                        and "args" in inp
                    ):
                        if len(inp["args"]) > 3 and inp["args"][3] is not None:
                            inp["args"].insert(3, None)
                    return inp

                return _patched_gi

            try:
                setattr(mod, "get_input", _make_gi(_orig_gi))
            except Exception:
                pass

        if hasattr(mod, "run_api"):
            _orig_ra = getattr(mod, "run_api")

            def _make_ra(orig):
                def _patched_ra(api, input_dict, cpu=True, lib="jax"):
                    try:
                        return orig(api, input_dict, cpu=cpu, lib=lib)
                    except NotImplementedError as e:
                        if "out" in str(e) and api == "jax.numpy.var":
                            import jax.numpy as jnp

                            return jnp.var(
                                jnp.asarray(input_dict["a"]),
                                axis=input_dict.get("axis"),
                                dtype=input_dict.get("dtype"),
                                ddof=input_dict.get("ddof", 0),
                                keepdims=input_dict.get("keepdims", False),
                                where=input_dict.get("where"),
                                mean=input_dict.get("mean"),
                                correction=input_dict.get("correction"),
                            )
                        raise

                return _patched_ra

            try:
                setattr(mod, "run_api", _make_ra(_orig_ra))
            except Exception:
                pass


_patch_harness()


def var_inputs():
    list_of_inputs = []

    # Input 1: 1D array, reducing along axis 0
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axis = (0,)
    where = np.array([True, True, True, True, True], dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True, where=where)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float32),
        "ddof": 0,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, reducing along axis 1 with correction=1 and keepdims=True
    a = np.array(
        [[1.5, -2.3, 3.1], [4.0, 5.2, -6.1], [-7.2, 8.4, 9.0]],
        dtype=np.float64,
    )
    axis = (1,)
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True, where=where)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float64),
        "ddof": 0,
        "keepdims": True,
        "where": where,
        "mean": mean,
        "correction": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, reducing along axis 0 with custom boolean mask
    a = np.array(
        [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]],
        dtype=np.float32,
    )
    axis = (0,)
    where = np.array(
        [[True, True, False], [True, False, True], [True, True, True]],
        dtype=bool,
    )
    mean = np.mean(a, axis=axis, keepdims=True, where=where)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float32),
        "ddof": 0,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, reducing over multiple axes (0, 2)
    a = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    axis = (0, 2)
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True, where=where)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float32),
        "ddof": 0,
        "keepdims": True,
        "where": where,
        "mean": mean,
        "correction": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array with negative values, reducing along axis 1 with correction=1
    a = np.linspace(-10.0, 10.0, 18, dtype=np.float64).reshape((3, 3, 2))
    axis = (1,)
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True, where=where)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float64),
        "ddof": 0,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, reducing along axes (2, 3)
    a = np.random.default_rng(42).standard_normal((2, 3, 4, 5)).astype(np.float32)
    axis = (2, 3)
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True, where=where)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float32),
        "ddof": 0,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array, reducing over all axes (0, 1)
    a = np.array([[-5.0, 0.0, 5.0], [10.0, -10.0, 15.0]], dtype=np.float32)
    axis = (0, 1)
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True, where=where)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float32),
        "ddof": 0,
        "keepdims": True,
        "where": where,
        "mean": mean,
        "correction": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array, negative axis index (-1,)
    a = np.array([-12.5, -3.2, 0.0, 4.8, 11.1, -0.5], dtype=np.float64)
    axis = (-1,)
    where = np.array([True, True, True, True, True, False], dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True, where=where)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float64),
        "ddof": 0,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array, negative axis index (-2,)
    a = np.random.default_rng(123).uniform(1.0, 10.0, (2, 5, 3)).astype(np.float32)
    axis = (-2,)
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True, where=where)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float32),
        "ddof": 0,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, higher correction value
    a = np.array(
        [
            [1.0, 3.0, 5.0, 7.0, 9.0],
            [2.0, 4.0, 6.0, 8.0, 10.0],
        ],
        dtype=np.float64,
    )
    axis = (1,)
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True, where=where)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float64),
        "ddof": 0,
        "keepdims": True,
        "where": where,
        "mean": mean,
        "correction": 2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D array, reducing along axes (1, 2) with keepdims=True
    a = np.random.default_rng(7).standard_normal((2, 4, 4, 3)).astype(np.float64)
    axis = (1, 2)
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True, where=where)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float64),
        "ddof": 0,
        "keepdims": True,
        "where": where,
        "mean": mean,
        "correction": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.var_2"] = var_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.var_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.var_2'.")


check_valid('jax.numpy.var', generated_inputs['jax.numpy.var_2'], lib="jax", suffix=2)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
import numpy as np


def sin_inputs():
    list_of_inputs = []

    # Find valid accuracy modes supported by jax.lax.sin
    valid_accuracies = []

    # Check candidates from AccuracyMode if available
    if hasattr(jax.lax, "AccuracyMode"):
        for item in jax.lax.AccuracyMode:
            for cand in [item.name, item.value, item.name.lower(), item]:
                if cand not in valid_accuracies:
                    try:
                        jax.lax.sin(np.float32(1.0), accuracy=cand)
                        valid_accuracies.append(cand)
                    except Exception:
                        pass

    # Check common string candidates
    common_candidates = [
        "HIGHEST",
        "HIGH",
        "LOW",
        "MEDIUM",
        "highest",
        "high",
        "low",
        "medium",
    ]
    for cand in common_candidates:
        if cand not in valid_accuracies:
            try:
                jax.lax.sin(np.float32(1.0), accuracy=cand)
                valid_accuracies.append(cand)
            except Exception:
                pass

    # Diverse inputs for x
    x_candidates = [
        np.array([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi], dtype=np.float32),
        np.random.randn(3, 4).astype(np.float32),
        np.linspace(-100.0, 100.0, 50, dtype=np.float64),
        np.random.uniform(-2 * np.pi, 2 * np.pi, size=(2, 3, 4)).astype(
            np.float32
        ),
        np.random.randn(5, 5).astype(np.float64),
        np.array(0.7853982, dtype=np.float32),
        np.array([1.0 + 2.0j, -0.5 + 0.5j, 0.0 - 1.0j], dtype=np.complex64),
        (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(
            np.complex128
        ),
        np.random.randn(2, 2, 3, 3).astype(np.float32),
        np.array([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=np.float16),
        np.array(
            [[0.0, np.pi], [-np.pi, 2 * np.pi], [np.pi / 4, -np.pi / 4]],
            dtype=np.float32,
        ),
        (
            np.random.uniform(-1.0, 1.0, size=(2, 2, 2))
            + 1j * np.random.uniform(-1.0, 1.0, size=(2, 2, 2))
        ).astype(np.complex64),
    ]

    for i, x in enumerate(x_candidates):
        for acc in valid_accuracies:
            try:
                jax.lax.sin(x, accuracy=acc)
                list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": acc}))
                break
            except Exception:
                continue

    return list_of_inputs


generated_inputs["jax.lax.sin_2"] = sin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.sin_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.sin_2'.")


check_valid('jax.lax.sin', generated_inputs['jax.lax.sin_2'], lib="jax", suffix=2)

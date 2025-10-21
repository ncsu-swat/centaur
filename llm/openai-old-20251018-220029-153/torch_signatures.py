signatures = {}
signatures["torch.DoubleStorage"] = {
    "args": {
        "size": "integer" # could be tuple
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.DoubleStorage_1"] = {
    "args": {
        "data": "list" # could be a tensor or list
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.ShortStorage"] = {
    "args": {
        "size": "integer",  # Could also be tuple, but integer is more common
    },
    "kwargs": {
        "dtype": "dtype",
        "device": "string", # Skipping device
    },
    "inner": {},
}
signatures["torch.abs"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # Could also be None
    },
    "inner": {}
}
signatures["torch.abs_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor" # Could also be None
    },
    "inner": {}
}
signatures["torch.absolute"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor" # Could also be None
    },
    "inner": {}
}
signatures["torch.acos"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # Could be None as well
    },
    "inner": {}
}
signatures["torch.acos_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor" # Could be None as well
    },
    "inner": {}
}
signatures["torch.acosh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.acosh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.add"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"  # Could also be a number (float/integer)
    },
    "kwargs": {
        "alpha": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.addbmm"] = {
    "args": {
        "input": "tensor",
        "batch1": "tensor",
        "batch2": "tensor"
    },
    "kwargs": {
        "beta": "float",  # Could be integer as well
        "alpha": "float", # Could be integer as well
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.addcdiv"] = {
    "args": {
        "input": "tensor",
        "tensor1": "tensor",
        "tensor2": "tensor"
    },
    "kwargs": {
        "value": "float",  # Could also be integer, but doc says real number for FloatTensor/DoubleTensor
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.addcmul"] = {
    "args": {
        "input": "tensor",
        "tensor1": "tensor",
        "tensor2": "tensor"
    },
    "kwargs": {
        "value": "float", # could be integer as well, but doc says real number for FloatTensor/DoubleTensor
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.addmm"] = {
    "args": {
        "input": "tensor",
        "mat1": "tensor",
        "mat2": "tensor"
    },
    "kwargs": {
        "out_dtype": "dtype",  # Could be None
        "beta": "float",
        "alpha": "float",
        "out": "tensor"  # Could be None
    },
    "inner": {}
}
signatures["torch.addmv"] = {
    "args": {
        "input": "tensor",
        "mat": "tensor",
        "vec": "tensor"
    },
    "kwargs": {
        "beta": "float",  # Could be number, but float seems more specific
        "alpha": "float",  # Could be number, but float seems more specific
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.addmv_"] = {
    "args": {
        "tensor": "tensor",
        "vec": "tensor",
        "rows": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.addr"] = {
    "args": {
        "input": "tensor",
        "vec1": "tensor",
        "vec2": "tensor"
    },
    "kwargs": {
        "beta": "float",
        "alpha": "float",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.adjoint"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.alias_copy"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.all_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}

signatures["torch.all_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"  # or tuple of integers
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.allclose"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "rtol": "float",
        "atol": "float",
        "equal_nan": "boolean"
    },
    "inner": {}
}
signatures["torch.amax"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",  # or tuple of integers
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tensor",
    },
    "inner": {},
}

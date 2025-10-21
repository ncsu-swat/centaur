signatures = {}
signatures["torch.DoubleStorage"] = {
    "args": {
        "size": "integer"  # Could also be a tuple of integers
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.ShortStorage"] = {
    "args": {
        "size": "integer" # Could also be tuple, but documentation implies integer
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.abs"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could also be None
    },
    "inner": {}
}
signatures["torch.abs_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # Could be None as well
    },
    "inner": {},
}
signatures["torch.absolute"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could also be None
    },
    "inner": {},
}
signatures["torch.acos"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could also be None
    },
    "inner": {}
}
signatures["torch.acos_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.acosh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # potentially could be None, but tensor is more specific
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
        "other": "tensor"  # Could also be "float" or "integer"
    },
    "kwargs": {
        "alpha": "float",
        "out": "tensor"
    },
    "inner": {},
}

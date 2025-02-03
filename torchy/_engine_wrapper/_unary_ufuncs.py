from ._utils import ArrayType, infer_engine_from_arrays

def log(a: ArrayType) -> ArrayType:
        engine = infer_engine_from_arrays(a)
        return engine.log(a)

def exp(a: ArrayType) -> ArrayType:
    engine = infer_engine_from_arrays(a)
    return engine.exp(a)

def sin(a: ArrayType) -> ArrayType:
    engine = infer_engine_from_arrays(a)
    return engine.sin(a)

def cos(a: ArrayType) -> ArrayType:
    engine = infer_engine_from_arrays(a)
    return engine.cos(a)

def tan(a: ArrayType) -> ArrayType:
    engine = infer_engine_from_arrays(a)
    return engine.tan(a)

def arcsin(a: ArrayType) -> ArrayType:
    engine = infer_engine_from_arrays(a)
    return engine.arcsin(a)

def arccos(a: ArrayType) -> ArrayType:
    engine = infer_engine_from_arrays(a)
    return engine.arccos(a)

def arctan(a: ArrayType) -> ArrayType:
    engine = infer_engine_from_arrays(a)
    return engine.arctan(a)

def sqrt(a: ArrayType) -> ArrayType:
    engine = infer_engine_from_arrays(a)
    return engine.sqrt(a)

def abs(a: ArrayType) -> ArrayType:
    engine = infer_engine_from_arrays(a)
    return engine.abs(a)

def sign(a: ArrayType) -> ArrayType:
    engine = infer_engine_from_arrays(a)
    return engine.sign(a)

def mean(a: ArrayType) -> ArrayType:
    engine = infer_engine_from_arrays(a)
    return engine.mean(a)

def sum(a: ArrayType, axis=None) -> ArrayType:
    engine = infer_engine_from_arrays(a)
    return engine.sum(a, axis=None)
from ._utils import ArrayType, infer_engine_from_array

def log(a: ArrayType) -> ArrayType:
        engine = infer_engine_from_array(a)
        return engine.log(a)

def exp(a: ArrayType) -> ArrayType:
    engine = infer_engine_from_array(a)
    return engine.exp(a)

def sin(a: ArrayType) -> ArrayType:
    engine = infer_engine_from_array(a)
    return engine.sin(a)

def cos(a: ArrayType) -> ArrayType:
    engine = infer_engine_from_array(a)
    return engine.cos(a)

def tan(a: ArrayType) -> ArrayType:
    engine = infer_engine_from_array(a)
    return engine.tan(a)

def arcsin(a: ArrayType) -> ArrayType:
    engine = infer_engine_from_array(a)
    return engine.arcsin(a)

def arccos(a: ArrayType) -> ArrayType:
    engine = infer_engine_from_array(a)
    return engine.arccos(a)

def arctan(a: ArrayType) -> ArrayType:
    engine = infer_engine_from_array(a)
    return engine.arctan(a)

def sqrt(a: ArrayType) -> ArrayType:
    engine = infer_engine_from_array(a)
    return engine.sqrt(a)

def abs(a: ArrayType) -> ArrayType:
    engine = infer_engine_from_array(a)
    return engine.abs(a)

def sign(a: ArrayType) -> ArrayType:
    engine = infer_engine_from_array(a)
    return engine.sign(a)
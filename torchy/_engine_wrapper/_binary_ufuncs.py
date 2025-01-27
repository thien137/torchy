from ._utils import ArrayType, infer_engine_from_array

def add(a: ArrayType, b: ArrayType) -> ArrayType:
    engine = infer_engine_from_array(a)
    return engine.add(a, b)

def subtract(a: ArrayType, b: ArrayType) -> ArrayType:
    engine = infer_engine_from_array(a)
    return engine.subtract(a, b)

def multiply(a: ArrayType, b: ArrayType) -> ArrayType:
    engine = infer_engine_from_array(a)
    return engine.multiply(a, b)

def divide(a: ArrayType, b: ArrayType) -> ArrayType:
    engine = infer_engine_from_array(a)
    return engine.divide(a, b)

def pow(a: ArrayType, b: ArrayType) -> ArrayType:
    engine = infer_engine_from_array(a)
    return engine.pow(a, b)

def maximum(a: ArrayType, b: ArrayType) -> ArrayType:
    engine = infer_engine_from_array(a)
    return engine.maximum(a, b)

def minimum(a: ArrayType, b: ArrayType) -> ArrayType:
    engine = infer_engine_from_array(a)
    return engine.minimum(a, b)

def equal(a: ArrayType, b: ArrayType) -> ArrayType:
    engine = infer_engine_from_array(a)
    return engine.equal(a, b)
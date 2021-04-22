import numpy as np


def db_five(min_value: float, max_value: float):
    a_max = max_value - 10
    aux = np.linspace(a_max-30, a_max, 4)
    actions = [min_value, *aux]
    return actions


def db_six(min_value: float, max_value: float):
    a_max = max_value - 10
    aux = np.linspace(a_max-40, a_max, 5)
    a_min = min_value if min_value < max_value-40 else -90
    actions = [a_min, *aux]
    return actions


def db_ten(min_value: float, max_value: float):
    aux = [max_value/2, max_value]
    aux2 = np.linspace(max_value-60, max_value-10, 7)
    a_min = min_value if min_value < max_value-40 else -90
    actions = [a_min, *aux2, *aux]
    return actions

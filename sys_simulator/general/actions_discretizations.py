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


def db_20():
    a1 = np.linspace(-60, -20, 10)
    a2 = np.linspace(-14, 0, 9)
    actions = [-90, *a1, *a2]
    return actions

def db_30():
    a0 = np.linspace(-90, -64, 10)
    a1 = np.linspace(-60, -20, 10)
    a2 = np.linspace(-14, 0, 10)
    actions = [*a0, *a1, *a2]
    return actions


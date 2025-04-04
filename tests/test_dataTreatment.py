import numpy as np
import os
from dataTreatment import load_data

file = "../data/Stocks/AAPL.csv"

def test_load_data_returns_numpy_array():
    # Es numpy_array
    data = load_data(file)
    assert isinstance(data, np.ndarray)

def test_no_nans_in_output():
    # No hay datos NaN
    data = load_data(file)
    assert not np.isnan(data).any()

def test_data_normalized():
    # Los datos estan entre 0 y 1
    data = load_data(file)
    assert np.all(data >= 0) and np.all(data <= 1)

def test_data_has_enough_rows():
    # No se devuelven muy pocas filas
    data = load_data(file)
    assert data.shape[0] > 50

def test_data_has_expected_columns_count():
    # Tiene el numero de columnas de entrada mas indicadores
    data = load_data(file)
    expected_cols = 6 + 8  # 6 originales + 8 indicadores tecnicos
    assert data.shape[1] == expected_cols

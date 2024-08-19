import numpy as np
from eigen_squared.eigenpairs.QR import QREigenSolver as QR


N = 5
b = np.random.randint(-2000, 2000, size=(N, N))
b_symm = (b + b.T) / 2
c = b_symm.T @ b_symm
np_q = abs(np.linalg.qr(b_symm)[0])

def test_QR_GS_Q():
    e2_out = abs(QR.decompose(b_symm, "GS")[0])
    assert np.allclose(e2_out, np_q), "QR GS Q does not match numpy output."

def test_QR_GS_R():
    e2_out = abs(QR.decompose(b_symm, "GS")[1])
    np_out = abs(np.linalg.qr(b_symm)[1])
    assert np.allclose(e2_out, np_out), "QR GS R does not match numpy output."

def test_QR_MGS_Q():
    e2_out = abs(QR.decompose(b_symm, "MGS")[0])
    assert np.allclose(e2_out, np_q), "QR MGS Q does not match numpy output."

def test_QR_MGS_R():
    e2_out = abs(QR.decompose(b_symm, "MGS")[1])
    np_out = abs(np.linalg.qr(b_symm)[1])
    assert np.allclose(e2_out, np_out), "QR MGS R does not match numpy output."

def test_QR_HR_Q():
    e2_out = abs(QR.decompose(b_symm, "HR")[0])
    assert np.allclose(e2_out, np_q), "QR HR Q does not match numpy output."

def test_QR_HR_R():
    e2_out = abs(QR.decompose(b_symm, "HR")[1])
    np_out = abs(np.linalg.qr(b_symm)[1])
    assert np.allclose(e2_out, np_out), "QR HR R does not match numpy output."

def test_QR_GR_Q():
    e2_out = abs(QR.decompose(b_symm, "GR")[0])
    assert np.allclose(e2_out, np_q), "QR GR Q does not match numpy output."

def test_QR_GR_R():
    e2_out = abs(QR.decompose(b_symm, "GR")[1])
    np_out = abs(np.linalg.qr(b_symm)[1])
    assert np.allclose(e2_out, np_out), "QR GR R does not match numpy output."

def test_cholesky_cc():
    e2_out = Cholesky.decompose(c, "CC")
    np_out = np.linalg.cholesky(c)
    print(np_out)

def test_unpivoted_Q_from_pivoted_Q():
    pivoted_QR = QR.decompose(b_symm, "MGS", True)
    unpivoted_QR = QR.unpivoted_Q_from_pivoted_Q(pivoted_QR[0], pivoted_QR[1])
    assert np.allclose(unpivoted_QR[0], pivoted_QR[0]), "Unpivoted Q does not match original Q."
    assert np.allclose(unpivoted_QR[1], pivoted_QR[1]), "R does not match original R."
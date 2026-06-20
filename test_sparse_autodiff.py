import numpy as np
from sparse_autodiff import (
    SparseMatrixCSR, Variable,
    sparse_add, sparse_matvec, sparse_transpose, sparse_slice,
    dense_add, sparse_dense_add, dense_sum, sparse_sum,
    linear
)


def test_sparse_matrix_basic():
    print("=== Test 1: SparseMatrixCSR 基本功能 ===")
    dense = np.array([
        [1.0, 0.0, 2.0],
        [0.0, 3.0, 0.0],
        [4.0, 0.0, 5.0]
    ])
    A = SparseMatrixCSR.from_dense(dense)
    assert A.shape == (3, 3)
    assert A.nnz == 5
    assert np.allclose(A.to_dense(), dense)
    print("  indptr:", A.indptr)
    print("  indices:", A.indices)
    print("  data:", A.data)
    print("  通过!")


def test_sparse_add():
    print("\n=== Test 2: 稀疏矩阵加法 ===")
    dense1 = np.array([
        [1.0, 0.0, 2.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 4.0]
    ])
    dense2 = np.array([
        [0.0, 1.0, 0.0],
        [2.0, 0.0, 3.0],
        [0.0, 4.0, 0.0]
    ])
    A = Variable(SparseMatrixCSR.from_dense(dense1), requires_grad=True, name="A")
    B = Variable(SparseMatrixCSR.from_dense(dense2), requires_grad=True, name="B")
    C = sparse_add(A, B)
    expected = dense1 + dense2
    assert np.allclose(C.value.to_dense(), expected)
    loss = sparse_sum(C)
    loss.backward()
    assert A.grad is not None
    assert B.grad is not None
    expected_grad_A = (dense1 != 0).astype(float)
    expected_grad_B = (dense2 != 0).astype(float)
    assert np.allclose(A.grad.to_dense(), expected_grad_A)
    assert np.allclose(B.grad.to_dense(), expected_grad_B)
    assert A.grad.nnz == A.value.nnz
    assert B.grad.nnz == B.value.nnz
    print("  C = A + B:\n", C.value.to_dense())
    print("  grad_A:\n", A.grad.to_dense())
    print("  grad_B:\n", B.grad.to_dense())
    print("  通过!")


def test_sparse_matvec_gradient():
    print("\n=== Test 3: 稀疏矩阵向量乘法及梯度 (y = A * x) ===")
    A_dense = np.array([
        [1.0, 0.0, 2.0],
        [0.0, 3.0, 0.0],
        [4.0, 0.0, 5.0]
    ])
    x = np.array([1.0, 2.0, 3.0])
    A = Variable(SparseMatrixCSR.from_dense(A_dense), requires_grad=True, name="A")
    x_var = Variable(x, requires_grad=True, name="x")
    y = sparse_matvec(A, x_var)
    expected_y = A_dense @ x
    assert np.allclose(y.value, expected_y)
    loss = dense_sum(y)
    loss.backward()
    expected_grad_A = np.tile(x, (A_dense.shape[0], 1)) * (A_dense != 0).astype(float)
    expected_grad_x = A_dense.sum(axis=0)
    print("  y = A * x:", y.value)
    print("  Expected y:", expected_y)
    print("  grad_A:\n", A.grad.to_dense())
    print("  Expected grad_A:\n", expected_grad_A)
    print("  grad_x:", x_var.grad)
    print("  Expected grad_x:", expected_grad_x)
    assert np.allclose(A.grad.to_dense(), expected_grad_A)
    assert np.allclose(x_var.grad, expected_grad_x)
    print("  通过!")


def test_linear_model():
    print("\n=== Test 4: 线性模型 y = A * x + b ===")
    np.random.seed(42)
    A_dense = np.zeros((4, 3))
    A_dense[0, 0] = 1.0
    A_dense[0, 2] = 2.0
    A_dense[1, 1] = 3.0
    A_dense[2, 0] = 4.0
    A_dense[3, 1] = 5.0
    A_dense[3, 2] = 6.0
    x = np.array([1.0, 2.0, 3.0])
    b = np.array([0.1, 0.2, 0.3, 0.4])
    A = Variable(SparseMatrixCSR.from_dense(A_dense), requires_grad=True, name="A")
    x_var = Variable(x, requires_grad=True, name="x")
    b_var = Variable(b, requires_grad=True, name="b")
    y = linear(A, x_var, b_var)
    expected_y = A_dense @ x + b
    assert np.allclose(y.value, expected_y)
    loss = dense_sum(y)
    loss.backward()
    expected_grad_A = np.tile(x, (A_dense.shape[0], 1)) * (A_dense != 0).astype(float)
    expected_grad_x = A_dense.sum(axis=0)
    expected_grad_b = np.ones_like(b)
    print("  y = A*x + b:", y.value)
    print("  Expected y:", expected_y)
    print("  grad_A:\n", A.grad.to_dense())
    print("  Expected grad_A:\n", expected_grad_A)
    print("  grad_x:", x_var.grad)
    print("  Expected grad_x:", expected_grad_x)
    print("  grad_b:", b_var.grad)
    print("  Expected grad_b:", expected_grad_b)
    assert np.allclose(A.grad.to_dense(), expected_grad_A)
    assert np.allclose(x_var.grad, expected_grad_x)
    assert np.allclose(b_var.grad, expected_grad_b)
    print("  通过!")


def test_sparse_transpose():
    print("\n=== Test 5: 稀疏矩阵转置 ===")
    A_dense = np.array([
        [1.0, 0.0, 2.0],
        [0.0, 3.0, 0.0]
    ])
    A = Variable(SparseMatrixCSR.from_dense(A_dense), requires_grad=True, name="A")
    AT = sparse_transpose(A)
    expected = A_dense.T
    assert np.allclose(AT.value.to_dense(), expected)
    loss = sparse_sum(AT)
    loss.backward()
    print("  A^T:\n", AT.value.to_dense())
    print("  grad_A:\n", A.grad.to_dense())
    assert np.allclose(A.grad.to_dense(), (A_dense != 0).astype(float))
    print("  通过!")


def test_sparse_slice():
    print("\n=== Test 6: 稀疏矩阵切片 ===")
    A_dense = np.array([
        [1.0, 0.0, 2.0],
        [0.0, 3.0, 0.0],
        [4.0, 0.0, 5.0],
        [0.0, 6.0, 0.0]
    ])
    A = Variable(SparseMatrixCSR.from_dense(A_dense), requires_grad=True, name="A")
    B = sparse_slice(A, slice(1, 3))
    expected = A_dense[1:3, :]
    assert np.allclose(B.value.to_dense(), expected)
    loss = sparse_sum(B)
    loss.backward()
    expected_grad_A = np.zeros_like(A_dense)
    expected_grad_A[1:3, :] = (A_dense[1:3, :] != 0).astype(float)
    print("  A[1:3, :]:\n", B.value.to_dense())
    print("  grad_A:\n", A.grad.to_dense())
    print("  Expected grad_A:\n", expected_grad_A)
    assert np.allclose(A.grad.to_dense(), expected_grad_A)
    print("  通过!")


def test_sparse_memory_efficiency():
    print("\n=== Test 7: 稀疏性验证（梯度只保留非零位置） ===")
    np.random.seed(42)
    n = 100
    sparsity = 0.95
    A_dense = np.zeros((n, n))
    nnz_per_row = 5
    for i in range(n):
        cols = np.random.choice(n, nnz_per_row, replace=False)
        vals = np.random.randn(nnz_per_row)
        A_dense[i, cols] = vals
    A_sparse = SparseMatrixCSR.from_dense(A_dense)
    print(f"  矩阵大小: {n}x{n}, 稠密元素数: {n*n}, 非零元素数: {A_sparse.nnz}")
    print(f"  稀疏率: {1 - A_sparse.nnz / (n*n):.4f}")
    A = Variable(A_sparse, requires_grad=True, name="A")
    x = Variable(np.random.randn(n), requires_grad=True, name="x")
    y = sparse_matvec(A, x)
    loss = dense_sum(y)
    loss.backward()
    assert A.grad.nnz == A_sparse.nnz, "梯度非零元素数应等于原矩阵非零元素数"
    print(f"  A.grad 非零元素数: {A.grad.nnz} (等于 A.nnz)")
    dense_mem = n * n * 8 / 1024
    sparse_mem = (A_sparse.nnz * 8 + (n + 1) * 8 + A_sparse.nnz * 8) / 1024
    print(f"  稠密存储内存: {dense_mem:.2f} KB")
    print(f"  稀疏存储内存: {sparse_mem:.2f} KB")
    print(f"  内存节省: {(1 - sparse_mem / dense_mem) * 100:.2f}%")
    print("  通过!")


def test_composite_expression():
    print("\n=== Test 8: 复合表达式梯度检验 ===")
    A_dense = np.array([
        [2.0, 0.0],
        [0.0, 3.0],
        [1.0, 0.0]
    ])
    B_dense = np.array([
        [0.0, 1.0],
        [2.0, 0.0],
        [0.0, 4.0]
    ])
    x = np.array([1.0, 2.0])
    b = np.array([0.5, 0.5, 0.5])
    A = Variable(SparseMatrixCSR.from_dense(A_dense), requires_grad=True, name="A")
    B = Variable(SparseMatrixCSR.from_dense(B_dense), requires_grad=True, name="B")
    x_var = Variable(x, requires_grad=True, name="x")
    b_var = Variable(b, requires_grad=True, name="b")
    C = sparse_add(A, B)
    Cx = sparse_matvec(C, x_var)
    y = dense_add(Cx, b_var)
    loss = dense_sum(y)
    loss.backward()
    expected_y = (A_dense + B_dense) @ x + b
    assert np.allclose(y.value, expected_y)
    expected_grad_x = (A_dense + B_dense).sum(axis=0)
    expected_grad_b = np.ones_like(b)
    expected_grad_A = np.tile(x, (A_dense.shape[0], 1)) * (A_dense != 0).astype(float)
    expected_grad_B = np.tile(x, (B_dense.shape[0], 1)) * (B_dense != 0).astype(float)
    print("  grad_A:\n", A.grad.to_dense())
    print("  Expected grad_A:\n", expected_grad_A)
    print("  grad_B:\n", B.grad.to_dense())
    print("  Expected grad_B:\n", expected_grad_B)
    print("  grad_x:", x_var.grad)
    print("  Expected grad_x:", expected_grad_x)
    assert np.allclose(A.grad.to_dense(), expected_grad_A)
    assert np.allclose(B.grad.to_dense(), expected_grad_B)
    assert np.allclose(x_var.grad, expected_grad_x)
    assert np.allclose(b_var.grad, expected_grad_b)
    print("  通过!")


def run_all_tests():
    print("=" * 60)
    print("稀疏矩阵自动微分库 - 测试套件")
    print("=" * 60)
    test_sparse_matrix_basic()
    test_sparse_add()
    test_sparse_matvec_gradient()
    test_linear_model()
    test_sparse_transpose()
    test_sparse_slice()
    test_sparse_memory_efficiency()
    test_composite_expression()
    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

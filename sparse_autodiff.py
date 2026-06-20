import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class SparseMatrixCSR:
    """
    CSR (Compressed Sparse Row) 格式稀疏矩阵
    - indptr: 行指针数组，长度为 nrows + 1，indptr[i] 表示第 i 行第一个非零元素在 data 中的索引
    - indices: 列索引数组，长度为 nnz，indices[k] 表示第 k 个非零元素所在的列
    - data: 非零元素值数组，长度为 nnz
    - shape: (nrows, ncols)
    """

    def __init__(self, indptr: np.ndarray, indices: np.ndarray, data: np.ndarray,
                 shape: Tuple[int, int]):
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.data = np.asarray(data, dtype=np.float64)
        self.shape = shape
        self.nrows, self.ncols = shape
        self.nnz = len(self.data)

    @classmethod
    def from_dense(cls, dense: np.ndarray) -> 'SparseMatrixCSR':
        rows, cols = np.nonzero(dense)
        data = dense[rows, cols]
        indptr = np.zeros(dense.shape[0] + 1, dtype=np.int64)
        for i in range(len(rows)):
            indptr[rows[i] + 1] += 1
        indptr = np.cumsum(indptr)
        return cls(indptr, cols, data, dense.shape)

    def to_dense(self) -> np.ndarray:
        dense = np.zeros(self.shape, dtype=np.float64)
        for i in range(self.nrows):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                dense[i, self.indices[k]] = self.data[k]
        return dense

    def __repr__(self) -> str:
        return (f"SparseMatrixCSR(shape={self.shape}, nnz={self.nnz})\n"
                f"  indptr={self.indptr}\n"
                f"  indices={self.indices}\n"
                f"  data={self.data})")

    @classmethod
    def eye(cls, n: int) -> 'SparseMatrixCSR':
        indptr = np.arange(n + 1, dtype=np.int64)
        indices = np.arange(n, dtype=np.int64)
        data = np.ones(n, dtype=np.float64)
        return cls(indptr, indices, data, (n, n))

    @classmethod
    def zeros(cls, shape: Tuple[int, int]) -> 'SparseMatrixCSR':
        indptr = np.zeros(shape[0] + 1, dtype=np.int64)
        indices = np.array([], dtype=np.int64)
        data = np.array([], dtype=np.float64)
        return cls(indptr, indices, data, shape)


class Variable:
    """
    自动微分变量，包装稀疏矩阵或稠密向量
    """
    _id_counter = 0

    def __init__(self, value, requires_grad: bool = False, name: Optional[str] = None):
        Variable._id_counter += 1
        self.id = Variable._id_counter
        self.name = name or f"var_{self.id}"
        self.value = value
        self.requires_grad = requires_grad
        self.grad = None
        self._creator: Optional['Function'] = None
        self._grad_fn = None

    def is_sparse(self) -> bool:
        return isinstance(self.value, SparseMatrixCSR)

    def is_dense(self) -> bool:
        return isinstance(self.value, np.ndarray)

    def backward(self, grad=None):
        if grad is None:
            if self.is_dense():
                grad = np.ones_like(self.value)
            else:
                grad_data = np.ones_like(self.value.data)
                grad = SparseMatrixCSR(
                    self.value.indptr.copy(),
                    self.value.indices.copy(),
                    grad_data,
                    self.value.shape
                )
        topo_order = []
        visited = set()

        def build_topo(node):
            if node.id in visited:
                return
            visited.add(node.id)
            if node._creator is not None:
                for input_var in node._creator.inputs:
                    build_topo(input_var)
            topo_order.append(node)

        build_topo(self)

        grad_map = {self.id: grad}

        for node in reversed(topo_order):
            if node._creator is not None:
                cur_grad = grad_map[node.id]
                grads = node._creator.backward(cur_grad)
                for input_var, input_grad in zip(node._creator.inputs, grads):
                    if input_var.requires_grad:
                        if input_var.id in grad_map:
                            grad_map[input_var.id] = _add_grads(grad_map[input_var.id], input_grad)
                        else:
                            grad_map[input_var.id] = input_grad
            if node.requires_grad:
                node.grad = grad_map.get(node.id)

    def zero_grad(self):
        self.grad = None

    def __repr__(self) -> str:
        type_str = "sparse" if self.is_sparse() else "dense"
        shape = self.value.shape
        return f"Variable(name={self.name}, type={type_str}, shape={shape}, requires_grad={self.requires_grad})"


def _add_grads(g1, g2):
    if isinstance(g1, SparseMatrixCSR) and isinstance(g2, SparseMatrixCSR):
        return _sparse_add(g1, g2)
    elif isinstance(g1, np.ndarray) and isinstance(g2, np.ndarray):
        return g1 + g2
    else:
        if isinstance(g1, SparseMatrixCSR):
            return g1.to_dense() + g2
        else:
            return g1 + g2.to_dense()


class Function:
    """
    运算节点基类
    """

    def __init__(self, inputs: List[Variable]):
        self.inputs = inputs
        self.output = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError


def _sparse_add(A: SparseMatrixCSR, B: SparseMatrixCSR) -> SparseMatrixCSR:
    assert A.shape == B.shape, "Shape mismatch for sparse add"
    nrows, ncols = A.shape
    new_data = []
    new_indices = []
    new_indptr = [0]
    for i in range(nrows):
        a_start, a_end = A.indptr[i], A.indptr[i + 1]
        b_start, b_end = B.indptr[i], B.indptr[i + 1]
        a_ptr, b_ptr = a_start, b_start
        while a_ptr < a_end and b_ptr < b_end:
            a_col = A.indices[a_ptr]
            b_col = B.indices[b_ptr]
            if a_col < b_col:
                new_indices.append(a_col)
                new_data.append(A.data[a_ptr])
                a_ptr += 1
            elif a_col > b_col:
                new_indices.append(b_col)
                new_data.append(B.data[b_ptr])
                b_ptr += 1
            else:
                new_indices.append(a_col)
                new_data.append(A.data[a_ptr] + B.data[b_ptr])
                a_ptr += 1
                b_ptr += 1
        while a_ptr < a_end:
            new_indices.append(A.indices[a_ptr])
            new_data.append(A.data[a_ptr])
            a_ptr += 1
        while b_ptr < b_end:
            new_indices.append(B.indices[b_ptr])
            new_data.append(B.data[b_ptr])
            b_ptr += 1
        new_indptr.append(len(new_data))
    return SparseMatrixCSR(
        np.array(new_indptr, dtype=np.int64),
        np.array(new_indices, dtype=np.int64),
        np.array(new_data, dtype=np.float64),
        A.shape
    )


def _sparse_matvec(A: SparseMatrixCSR, x: np.ndarray) -> np.ndarray:
    assert A.ncols == x.shape[0], "Dimension mismatch for matvec"
    result = np.zeros(A.nrows, dtype=np.float64)
    for i in range(A.nrows):
        for k in range(A.indptr[i], A.indptr[i + 1]):
            result[i] += A.data[k] * x[A.indices[k]]
    return result


def _sparse_transpose(A: SparseMatrixCSR) -> SparseMatrixCSR:
    nrows, ncols = A.shape
    new_indptr = np.zeros(ncols + 1, dtype=np.int64)
    for col in A.indices:
        new_indptr[col + 1] += 1
    new_indptr = np.cumsum(new_indptr)
    new_indices = np.zeros(A.nnz, dtype=np.int64)
    new_data = np.zeros(A.nnz, dtype=np.float64)
    counter = new_indptr.copy()
    for i in range(nrows):
        for k in range(A.indptr[i], A.indptr[i + 1]):
            col = A.indices[k]
            pos = counter[col]
            new_indices[pos] = i
            new_data[pos] = A.data[k]
            counter[col] += 1
    return SparseMatrixCSR(new_indptr, new_indices, new_data, (ncols, nrows))


def _sparse_slice(A: SparseMatrixCSR, row_slice: slice) -> SparseMatrixCSR:
    start, stop, step = row_slice.indices(A.nrows)
    assert step == 1, "Only step=1 is supported for slicing"
    new_nrows = stop - start
    new_indptr = np.zeros(new_nrows + 1, dtype=np.int64)
    new_indices_list = []
    new_data_list = []
    for new_i, i in enumerate(range(start, stop)):
        for k in range(A.indptr[i], A.indptr[i + 1]):
            new_indices_list.append(A.indices[k])
            new_data_list.append(A.data[k])
        new_indptr[new_i + 1] = len(new_data_list)
    return SparseMatrixCSR(
        new_indptr,
        np.array(new_indices_list, dtype=np.int64),
        np.array(new_data_list, dtype=np.float64),
        (new_nrows, A.ncols)
    )


def _outer_add_sparse_dense(A: SparseMatrixCSR, b: np.ndarray) -> SparseMatrixCSR:
    assert A.nrows == b.shape[0], "Shape mismatch for outer add"
    new_data = np.zeros_like(A.data)
    b_2d = b.reshape(-1, 1)
    for i in range(A.nrows):
        for k in range(A.indptr[i], A.indptr[i + 1]):
            new_data[k] = A.data[k] + b_2d[i, 0]
    return SparseMatrixCSR(A.indptr.copy(), A.indices.copy(), new_data, A.shape)


def _extract_grad_by_pattern(source: SparseMatrixCSR, target_pattern: SparseMatrixCSR) -> SparseMatrixCSR:
    source_dict = {}
    for i in range(source.nrows):
        for k in range(source.indptr[i], source.indptr[i + 1]):
            source_dict[(i, source.indices[k])] = source.data[k]
    new_data = np.zeros(target_pattern.nnz, dtype=np.float64)
    for i in range(target_pattern.nrows):
        for k in range(target_pattern.indptr[i], target_pattern.indptr[i + 1]):
            col = target_pattern.indices[k]
            new_data[k] = source_dict.get((i, col), 0.0)
    return SparseMatrixCSR(target_pattern.indptr.copy(), target_pattern.indices.copy(), new_data, target_pattern.shape)


class SparseAdd(Function):
    def __init__(self, inputs: List[Variable]):
        super().__init__(inputs)

    def forward(self) -> Variable:
        A, B = self.inputs[0].value, self.inputs[1].value
        self.A_pattern = A
        self.B_pattern = B
        result = _sparse_add(A, B)
        return Variable(result, requires_grad=any(v.requires_grad for v in self.inputs))

    def backward(self, grad_output):
        grad_A = _extract_grad_by_pattern(grad_output, self.A_pattern)
        grad_B = _extract_grad_by_pattern(grad_output, self.B_pattern)
        return [grad_A, grad_B]


def sparse_add(A: Variable, B: Variable) -> Variable:
    assert A.is_sparse() and B.is_sparse()
    func = SparseAdd([A, B])
    out = func.forward()
    out._creator = func
    return out


class SparseMatVec(Function):
    def __init__(self, inputs: List[Variable]):
        super().__init__(inputs)

    def forward(self) -> Variable:
        A = self.inputs[0].value
        x = self.inputs[1].value
        result = _sparse_matvec(A, x)
        return Variable(result, requires_grad=any(v.requires_grad for v in self.inputs))

    def backward(self, grad_output):
        A_var, x_var = self.inputs
        A = A_var.value
        x = x_var.value

        grads = []
        if A_var.requires_grad:
            grad_A_indptr = A.indptr.copy()
            grad_A_indices = A.indices.copy()
            grad_A_data = np.zeros_like(A.data)
            for i in range(A.nrows):
                for k in range(A.indptr[i], A.indptr[i + 1]):
                    grad_A_data[k] = grad_output[i] * x[A.indices[k]]
            grads.append(SparseMatrixCSR(grad_A_indptr, grad_A_indices, grad_A_data, A.shape))
        else:
            grads.append(None)

        if x_var.requires_grad:
            grad_x = np.zeros(A.ncols, dtype=np.float64)
            for i in range(A.nrows):
                for k in range(A.indptr[i], A.indptr[i + 1]):
                    grad_x[A.indices[k]] += grad_output[i] * A.data[k]
            grads.append(grad_x)
        else:
            grads.append(None)

        return grads


def sparse_matvec(A: Variable, x: Variable) -> Variable:
    assert A.is_sparse() and x.is_dense()
    func = SparseMatVec([A, x])
    out = func.forward()
    out._creator = func
    return out


class SparseTranspose(Function):
    def __init__(self, inputs: List[Variable]):
        super().__init__(inputs)

    def forward(self) -> Variable:
        A = self.inputs[0].value
        result = _sparse_transpose(A)
        return Variable(result, requires_grad=self.inputs[0].requires_grad)

    def backward(self, grad_output):
        grad_input = _sparse_transpose(grad_output)
        return [grad_input]


def sparse_transpose(A: Variable) -> Variable:
    assert A.is_sparse()
    func = SparseTranspose([A])
    out = func.forward()
    out._creator = func
    return out


class SparseSlice(Function):
    def __init__(self, inputs: List[Variable], row_slice: slice):
        super().__init__(inputs)
        self.row_slice = row_slice

    def forward(self) -> Variable:
        A = self.inputs[0].value
        result = _sparse_slice(A, self.row_slice)
        return Variable(result, requires_grad=self.inputs[0].requires_grad)

    def backward(self, grad_output):
        A = self.inputs[0].value
        start, stop, step = self.row_slice.indices(A.nrows)
        full_grad_indptr = np.zeros(A.nrows + 1, dtype=np.int64)
        full_grad_indices = np.array([], dtype=np.int64)
        full_grad_data = np.array([], dtype=np.float64)
        for i in range(A.nrows):
            if start <= i < stop:
                local_i = i - start
                for k in range(grad_output.indptr[local_i], grad_output.indptr[local_i + 1]):
                    full_grad_indices = np.append(full_grad_indices, grad_output.indices[k])
                    full_grad_data = np.append(full_grad_data, grad_output.data[k])
            full_grad_indptr[i + 1] = len(full_grad_data)
        full_grad = SparseMatrixCSR(full_grad_indptr, full_grad_indices, full_grad_data, A.shape)
        return [full_grad]


def sparse_slice(A: Variable, row_slice: slice) -> Variable:
    assert A.is_sparse()
    func = SparseSlice([A], row_slice)
    out = func.forward()
    out._creator = func
    return out


class DenseAdd(Function):
    def __init__(self, inputs: List[Variable]):
        super().__init__(inputs)

    def forward(self) -> Variable:
        a, b = self.inputs[0].value, self.inputs[1].value
        result = a + b
        return Variable(result, requires_grad=any(v.requires_grad for v in self.inputs))

    def backward(self, grad_output):
        return [grad_output, grad_output]


def dense_add(a: Variable, b: Variable) -> Variable:
    assert a.is_dense() and b.is_dense()
    func = DenseAdd([a, b])
    out = func.forward()
    out._creator = func
    return out


class SparseDenseAdd(Function):
    def __init__(self, inputs: List[Variable]):
        super().__init__(inputs)

    def forward(self) -> Variable:
        A = self.inputs[0].value
        b = self.inputs[1].value
        result = _outer_add_sparse_dense(A, b)
        return Variable(result, requires_grad=any(v.requires_grad for v in self.inputs))

    def backward(self, grad_output):
        grads = []
        if self.inputs[0].requires_grad:
            grads.append(grad_output)
        else:
            grads.append(None)

        if self.inputs[1].requires_grad:
            A = self.inputs[0].value
            grad_b = np.zeros(A.nrows, dtype=np.float64)
            for i in range(A.nrows):
                row_sum = 0.0
                for k in range(grad_output.indptr[i], grad_output.indptr[i + 1]):
                    row_sum += grad_output.data[k]
                grad_b[i] = row_sum
            grads.append(grad_b)
        else:
            grads.append(None)

        return grads


def sparse_dense_add(A: Variable, b: Variable) -> Variable:
    assert A.is_sparse() and b.is_dense()
    func = SparseDenseAdd([A, b])
    out = func.forward()
    out._creator = func
    return out


class DenseSum(Function):
    def __init__(self, inputs: List[Variable]):
        super().__init__(inputs)

    def forward(self) -> Variable:
        x = self.inputs[0].value
        result = np.array([x.sum()], dtype=np.float64)
        return Variable(result, requires_grad=self.inputs[0].requires_grad)

    def backward(self, grad_output):
        x = self.inputs[0].value
        grad = np.ones_like(x) * grad_output[0]
        return [grad]


def dense_sum(x: Variable) -> Variable:
    assert x.is_dense()
    func = DenseSum([x])
    out = func.forward()
    out._creator = func
    return out


class SparseSum(Function):
    def __init__(self, inputs: List[Variable]):
        super().__init__(inputs)

    def forward(self) -> Variable:
        A = self.inputs[0].value
        result = np.array([A.data.sum()], dtype=np.float64)
        return Variable(result, requires_grad=self.inputs[0].requires_grad)

    def backward(self, grad_output):
        A = self.inputs[0].value
        grad_data = np.ones_like(A.data) * grad_output[0]
        grad = SparseMatrixCSR(A.indptr.copy(), A.indices.copy(), grad_data, A.shape)
        return [grad]


def sparse_sum(A: Variable) -> Variable:
    assert A.is_sparse()
    func = SparseSum([A])
    out = func.forward()
    out._creator = func
    return out


def linear(A: Variable, x: Variable, b: Variable) -> Variable:
    Ax = sparse_matvec(A, x)
    result = dense_add(Ax, b)
    return result

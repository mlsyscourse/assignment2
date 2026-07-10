import argparse

import numpy as np
import tvm
from tvm import s_tir
from tvm.runtime import cuda, empty, tensor

from gemm_relu_add import K, M, N, manual_schedule
from trace_submission import apply_trace

np.random.seed(0)


def build_sch(sch: s_tir.Schedule) -> tvm.runtime.Module:
    return tvm.build(sch.mod, target="cuda")


def test_numerical_correctness(sch: s_tir.Schedule, num_rounds: int = 5):
    f = build_sch(sch)

    for i in range(num_rounds):
        A_np = np.random.uniform(-1, 1, size=(M, K)).astype("float32")
        B_np = np.random.uniform(-1, 1, size=(K, N)).astype("float32")
        C_np = np.random.uniform(-1, 1, size=(M, N)).astype("float32")

        D_std = np.maximum(A_np @ B_np, 0) + C_np

        device = cuda()
        A_tvm = tensor(A_np, device=device)
        B_tvm = tensor(B_np, device=device)
        C_tvm = tensor(C_np, device=device)
        D_tvm = tensor(np.zeros((M, N), dtype="float32"), device=device)
        f(A_tvm, B_tvm, C_tvm, D_tvm)
        np.testing.assert_allclose(D_tvm.numpy(), D_std, rtol=1e-4, atol=1e-4)
        print(f"Passing test round {i}...")
    print("Passed all tests.")


def evaluate_execution_time(sch: s_tir.Schedule):
    f = build_sch(sch)

    device = cuda()
    A_tvm = empty((M, K), "float32", device)
    B_tvm = empty((K, N), "float32", device)
    C_tvm = empty((M, N), "float32", device)
    D_tvm = empty((M, N), "float32", device)

    t = f.time_evaluator(f.entry_name, device, number=3, repeat=10, min_repeat_ms=100)(
        A_tvm, B_tvm, C_tvm, D_tvm
    ).mean
    print("Execution time: %.2f ms" % (t * 1e3))


def evaluate_naive_func_execution_time():
    from gemm_relu_add import gemm_relu_add

    sch = s_tir.Schedule(gemm_relu_add)
    gemm_block = sch.get_sblock("gemm")
    i, j, _ = sch.get_loops(gemm_block)
    io, ii = sch.split(i, [None, 32])
    jo, ji = sch.split(j, [None, 32])
    sch.bind(io, "blockIdx.x")
    sch.bind(jo, "blockIdx.y")
    sch.bind(ii, "threadIdx.x")
    sch.bind(ji, "threadIdx.y")
    relu_block = sch.get_sblock("relu")
    sch.reverse_compute_at(relu_block, ji)
    add_block = sch.get_sblock("add")
    sch.reverse_compute_inline(add_block)
    sch.set_scope(gemm_block, 0, "local")
    # Uncomment the line below to check the naive function.
    # sch.show()

    f = build_sch(sch)

    device = cuda()
    A_tvm = empty((M, K), "float32", device)
    B_tvm = empty((K, N), "float32", device)
    C_tvm = empty((M, N), "float32", device)
    D_tvm = empty((M, N), "float32", device)

    t = f.time_evaluator(f.entry_name, device, number=3, repeat=10, min_repeat_ms=100)(
        A_tvm, B_tvm, C_tvm, D_tvm
    ).mean
    print("Naive function execution time: %.2f ms" % (t * 1e3))


def show_cuda(sch: s_tir.Schedule):
    f = build_sch(sch)
    print(f.imports[0].inspect_source())


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--test", action="store_true")
    args.add_argument("--evaluate-manual", action="store_true")
    args.add_argument("--evaluate-tuned", action="store_true")
    args.add_argument("--evaluate-naive", action="store_true")
    args.add_argument("--show-cuda", action="store_true")
    parsed = args.parse_args()

    if parsed.test:
        sch = manual_schedule()
        test_numerical_correctness(sch)
    if parsed.evaluate_manual:
        sch = manual_schedule()
        evaluate_execution_time(sch)
    if parsed.evaluate_tuned:
        from gemm_relu_add import gemm_relu_add

        sch = s_tir.Schedule(gemm_relu_add)
        apply_trace(sch)
        evaluate_execution_time(sch)
    if parsed.evaluate_naive:
        evaluate_naive_func_execution_time()
    if parsed.show_cuda:
        sch = manual_schedule()
        show_cuda(sch)

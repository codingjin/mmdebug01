import os
import sys
import numpy as np
import tvm
from tvm import te, auto_scheduler
import time

def matmul_gflops(input_shape, exe_time):
    M, N, K = input_shape
    flops = 2.0 * M * N * K
    gflops = flops / exe_time / 1e9
    return gflops

@auto_scheduler.register_workload
def matmul_add(M, N, K, dtype):
    A = te.placeholder((M, K), name = "A", dtype = dtype)
    B = te.placeholder((K, N), name = "B", dtype = dtype)
    C = te.placeholder((M, N), name = "C", dtype = dtype)

    k = te.reduce_axis((0, K), name = "k")
    matmul = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis = k),
        name = "matmul",
        attrs = {"layout_free_placeholders": [B]},
    )

    out = te.compute((M, N), lambda i, j: matmul[i, j] + C[i, j], name = "out")

    return [A, B, C, out]


def main():
    argv = sys.argv
    #print(len(argv))
    if len(argv) != 5:
        print("Invalid input!")
        print("python mm_ansor.py <THREADNUM> M N K!")
        exit(1)
    
    if argv[1] != '0':
        os.environ["TVM_NUM_THREADS"] = argv[1]
    
    M = int(argv[2])
    N = int(argv[3])
    K = int(argv[4])
    print("ThreadNum=", argv[1], " M=", M, " N=", N, " K=", K)
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    task = tvm.auto_scheduler.SearchTask(func=matmul_add, args=(M, N, K, "float32"), target=target)

    # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)

    log_file = "cpu_matmuladd_TN"+argv[1]+"_M"+str(M)+"_N"+str(N)+"_K"+str(K)+".json"
    start_time = int(time.time())
    csv_file_path = log_file.replace('.json', '.csv')

    # write the start time to the csv file
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_file.write(f"start_time:{str(start_time)}\n")

    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=1000,
        runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=0,
    )
    
    combined_res_file = 'results_TN'+argv[1]+"_M"+str(M)+"_N"+str(N)+"_K"+str(K)+".txt"
    # Run auto-tuning (search)
    task.tune(tune_option)

    # Apply the best schedule
    try:
        sch, args = task.apply_best(log_file)
        
        func = tvm.build(sch, args, target)
        
        np.random.seed(149)
        a_np = np.random.uniform(size=(M, K)).astype(np.float32)
        b_np = np.random.uniform(size=(K, N)).astype(np.float32)
        c_np = np.random.uniform(size=(M, N)).astype(np.float32)
        out_np = a_np.dot(b_np) + c_np

        dev = tvm.cpu()
        a_tvm = tvm.nd.array(a_np, device=dev)
        b_tvm = tvm.nd.array(b_np, device=dev)
        c_tvm = tvm.nd.array(c_np, device=dev)
        out_tvm = tvm.nd.empty(out_np.shape, device=dev)
        func(a_tvm, b_tvm, c_tvm, out_tvm)
        
        evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
        # Check results
        np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)
            
        # in seconds
        input_shape = (M, N, K)
        gflops = matmul_gflops(input_shape, np.median(evaluator(a_tvm, b_tvm, c_tvm, out_tvm).results))
            
        # write the gflops to the csv file
        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_file.write(f"gflops:{str(gflops)}\n")
        
        with open(combined_res_file, 'a') as f:
            f.write(f"Problem: TN:{argv[1]} M:{M} N:{N} K:{K} GFLOPS:{gflops}\n")
    except Exception as e:
        print(e)
        with open(combined_res_file, 'a') as f:
            f.write(f"Problem: TN:{argv[1]} M:{M} N:{N} K:{K} GFLOPS:0, Can't find any valid schedule\n")


if __name__ == "__main__":
    main()
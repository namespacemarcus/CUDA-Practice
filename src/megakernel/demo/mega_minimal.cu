#include <cuda_runtime.h>
#include <torch/extension.h>

struct Task {
    int type;    // 0: gemm_a, 1: gemm_b, -1: sentinel exit
    int dep_cnt; // 剩余前置依赖数量
    int succ[4]; // 后继任务id
    float *A;
    float *B;
    float *C;
    int M, N, K;
};

__device__ void simple_gemm(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.x * 32 + threadIdx.x;
    int col = blockIdx.y * 32 + threadIdx.y;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void megakernel(Task *task_list, int *dep_count, int num_task) {
    int block_id = blockIdx.x;

    // persistent 常驻循环， kernel launch后不退出
    while (true) {
        int tid{-1};
        // 轮询找就绪任务：dep_count == 0，原子抢占标记为 -1（占用）
        for (int i = 0; i < num_task; ++i) {
            int old = atomicCAS(&dep_count[i], 0, -1);
            if (old == 0) {
                tid = i;
                break;
            }
        }

        if (tid == -1) {
            __nanosleep(100);
            continue;
        }

        Task t = task_list[tid];
        if (t.type == -1) {
            break;
        }

        simple_gemm(t.A, t.B, t.C, t.M, t.N, t.K);

        for (int j = 0; j < 4; ++j) {
            int s = t.succ[j];
            if (s < 0) {
                break;
            }
            atomicSub(&dep_count[s], 1);
        }
    }
}

torch::Tensor run_megakernel(torch::Tensor A, torch::Tensor B, torch::Tensor C1,
                             torch::Tensor D, torch::Tensor C2,
                             torch::Tensor dep_host) {
    A = A.to(torch::kFloat32).to(torch::kCUDA);
    B = B.to(torch::kFloat32).to(torch::kCUDA);
    C1 = C1.to(torch::kFloat32).to(torch::kCUDA);
    D = D.to(torch::kFloat32).to(torch::kCUDA);
    C2 = C2.to(torch::kFloat32).to(torch::kCUDA);

    Task task_host[3];
    // Task0: gemm A*B -> C1，后继是 task1
    task_host[0] = {0,
                    0,
                    {1, -1, -1, -1},
                    (float *)A.data_ptr(),
                    (float *)B.data_ptr(),
                    (float *)C1.data_ptr(),
                    128,
                    128,
                    128};
    // Task1: gemm C1*D -> C2，后继是 sentinel task2
    task_host[1] = {0,
                    1,
                    {2, -1, -1, -1},
                    (float *)C1.data_ptr(),
                    (float *)D.data_ptr(),
                    (float *)C2.data_ptr(),
                    128,
                    128,
                    128};
    // Task2: sentinel exit
    task_host[2] = {-1, 1, {-1, -1, -1, -1}, nullptr, nullptr, nullptr, 0,
                    0,  0};

    Task *task_dev;
    int *dep_dev;
    cudaMalloc(&task_dev, sizeof(Task) * 3);
    cudaMalloc(&dep_dev, sizeof(int) * 3);
    cudaMemcpy(task_dev, task_host, sizeof(Task) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(dep_dev, dep_host.data_ptr(), sizeof(int) * 3,
               cudaMemcpyHostToDevice);

    // 只 launch 一次！！！重点：常驻 megakernel，不是launch多次
    dim3 grid(16);
    dim3 block(32 * 32);
    megakernel<<<grid, block>>>(task_dev, dep_dev, 3);
    cudaDeviceSynchronize();

    cudaFree(task_dev);
    cudaFree(dep_dev);
    return C2;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run_megakernel, "minimal persistent megakernel");
}

#include <random>
#include <sys/time.h>
#include <omp.h>
#include <cassert>
#include <cmath>

inline double get_clock_us()
{
    struct timeval clock;
    gettimeofday(&clock, NULL);
    return 1000000.0 * clock.tv_sec + clock.tv_usec;
}

void tensor_contraction_naive(int S, int L, int M, int I, int J, const float *A, const float *B, float *C)
{
    #pragma omp parallel for
    for (int l = 0; l < L; ++l) {
        for (int m = 0; m < M; ++m) {
            for (int i = 0; i < I; ++i) {
                for (int j = 0; j < J; ++j) {
                    double res = 0;
                    for (int s = 0; s < S; ++s) {
                        res += (double)A[s * L * I + l * I + i] * B[s * M * J + m * J + j];
                    }
                    C[l * M * I * J + m * I * J + i * J + j] = res;
                }
            }
        }
    }
}

void tensor_contraction(int S, int L, int M, int I, int J, const float *A, const float *B, float *C)
{
    // 分块大小，针对缓存优化
    const int L_BLOCK = 32;
    const int M_BLOCK = 32;
    const int I_BLOCK = 16;
    const int J_BLOCK = 16;
    
    // 初始化C为0
    #pragma omp parallel for
    for (int idx = 0; idx < L * M * I * J; ++idx) {
        C[idx] = 0;
    }
    
    // 外层循环按块划分
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int lb = 0; lb < L; lb += L_BLOCK) {
        for (int mb = 0; mb < M; mb += M_BLOCK) {
            int l_end = (lb + L_BLOCK < L) ? (lb + L_BLOCK) : L;
            int m_end = (mb + M_BLOCK < M) ? (mb + M_BLOCK) : M;
            
            for (int ib = 0; ib < I; ib += I_BLOCK) {
                for (int jb = 0; jb < J; jb += J_BLOCK) {
                    int i_end = (ib + I_BLOCK < I) ? (ib + I_BLOCK) : I;
                    int j_end = (jb + J_BLOCK < J) ? (jb + J_BLOCK) : J;
                    
                    // 对S维度进行累加
                    for (int s = 0; s < S; ++s) {
                        // 块内循环
                        for (int l = lb; l < l_end; ++l) {
                            for (int m = mb; m < m_end; ++m) {
                                for (int i = ib; i < i_end; ++i) {
                                    // 预取A的值
                                    float a_val = A[s * L * I + l * I + i];
                                    
                                    // J维度向量化
                                    int j = jb;
                                    // 向量化主循环 - 每次处理4个元素
                                    for (; j + 3 < j_end; j += 4) {
                                        int b_idx = s * M * J + m * J + j;
                                        int c_idx = l * M * I * J + m * I * J + i * J + j;
                                        
                                        float b0 = B[b_idx];
                                        float b1 = B[b_idx + 1];
                                        float b2 = B[b_idx + 2];
                                        float b3 = B[b_idx + 3];
                                        
                                        C[c_idx] += a_val * b0;
                                        C[c_idx + 1] += a_val * b1;
                                        C[c_idx + 2] += a_val * b2;
                                        C[c_idx + 3] += a_val * b3;
                                    }
                                    // 处理剩余元素
                                    for (; j < j_end; ++j) {
                                        int b_idx = s * M * J + m * J + j;
                                        int c_idx = l * M * I * J + m * I * J + i * J + j;
                                        C[c_idx] += a_val * B[b_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void tensor_contraction_test(int S, int L, int M, int I, int J)
{
    static constexpr int test_times = 10;
    float *A = (float *)aligned_alloc(64, S * L * I * test_times * sizeof(float));
    float *B = (float *)aligned_alloc(64, S * M * J * test_times * sizeof(float));
    float *C = (float *)aligned_alloc(64, L * M * I * J * test_times * sizeof(float));
    float *C_std = (float *)aligned_alloc(64, L * M * I * J * sizeof(float));

    #pragma omp parallel
    {
        std::mt19937 rnd(rand() ^ omp_get_thread_num());
        std::normal_distribution<> std_norm;
        #pragma omp for
        for (int i = 0; i < S * L * I * test_times; ++i) {
            A[i] = std_norm(rnd) / sqrt(S);
        }
        #pragma omp for
        for (int i = 0; i < S * M * J * test_times; ++i) {
            B[i] = std_norm(rnd) / sqrt(S);
        }
        #pragma omp for
        for (int i = 0; i < L * M * I * J * test_times; ++i) {
            C[i] = 0;
        }
    }

    tensor_contraction_naive(S, L, M, I, J, A, B, C_std);
    tensor_contraction(S, L, M, I, J, A, B, C);

    double max_diff = 0;
    for (int i = 0; i < L * M * I * J; ++i) {
        double diff = (double)C_std[i] - C[i];
        max_diff = std::max(max_diff, std::abs(diff));
    }

    double begin_time = get_clock_us();
    for (int i = 0; i < test_times; ++i) {
        tensor_contraction(S, L, M, I, J, A + i * S * L * I, B + i * S * M * J, C + i * L * M * I * J);
    }
    double end_time = get_clock_us();

    double avg_time = (end_time - begin_time) / test_times;
    printf("avg_time = %lf ms\n", avg_time / 1e3);

    free(A);
    free(B);
    free(C);
    free(C_std);

    if (max_diff < 1e-9) {
        printf("PASSED");
    } else {
        printf("FAILED!!! max_diff = %lf\n", max_diff);
    }
}

int main()
{
    srand(time(NULL));
    tensor_contraction_test(256, 256, 256, 32, 32);
    return 0;
}
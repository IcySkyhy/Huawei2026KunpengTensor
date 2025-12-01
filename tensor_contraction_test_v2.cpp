#include <random>
#include <sys/time.h>
#include <omp.h>
#include <cassert>
#include <cmath>
#include <cstring>

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

// 高度优化版本 - 使用更好的内存访问模式和分块
void tensor_contraction(int S, int L, int M, int I, int J, const float *A, const float *B, float *C)
{
    // 针对鲲鹏架构的分块参数
    const int S_BLOCK = 32;  // S维度分块
    const int L_BLOCK = 64;  // L维度分块
    const int M_BLOCK = 64;  // M维度分块
    const int IJ_BLOCK = 32; // I*J维度合并处理
    
    // 初始化C为0
    memset(C, 0, sizeof(float) * L * M * I * J);
    
    // 外层并行化L和M维度
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int lb = 0; lb < L; lb += L_BLOCK) {
        for (int mb = 0; mb < M; mb += M_BLOCK) {
            int l_end = std::min(lb + L_BLOCK, L);
            int m_end = std::min(mb + M_BLOCK, M);
            
            // 对S维度分块处理，提高缓存命中率
            for (int sb = 0; sb < S; sb += S_BLOCK) {
                int s_end = std::min(sb + S_BLOCK, S);
                
                // 处理L块
                for (int l = lb; l < l_end; ++l) {
                    // 处理M块
                    for (int m = mb; m < m_end; ++m) {
                        // 合并I和J循环以提高缓存效率
                        for (int ij = 0; ij < I * J; ij += IJ_BLOCK) {
                            int ij_end = std::min(ij + IJ_BLOCK, I * J);
                            
                            // S维度累加
                            for (int s = sb; s < s_end; ++s) {
                                // 向量化内层循环
                                for (int idx = ij; idx < ij_end; idx += 8) {
                                    int i = idx / J;
                                    int j_start = idx % J;
                                    
                                    if (i < I) {
                                        float a_val = A[s * L * I + l * I + i];
                                        int c_base = l * M * I * J + m * I * J + i * J;
                                        int b_base = s * M * J + m * J;
                                        
                                        // 展开循环处理8个元素
                                        int j = j_start;
                                        int j_max = std::min(J, j_start + (ij_end - idx));
                                        
                                        // 手动展开以帮助编译器向量化
                                        for (; j + 7 < j_max; j += 8) {
                                            C[c_base + j + 0] += a_val * B[b_base + j + 0];
                                            C[c_base + j + 1] += a_val * B[b_base + j + 1];
                                            C[c_base + j + 2] += a_val * B[b_base + j + 2];
                                            C[c_base + j + 3] += a_val * B[b_base + j + 3];
                                            C[c_base + j + 4] += a_val * B[b_base + j + 4];
                                            C[c_base + j + 5] += a_val * B[b_base + j + 5];
                                            C[c_base + j + 6] += a_val * B[b_base + j + 6];
                                            C[c_base + j + 7] += a_val * B[b_base + j + 7];
                                        }
                                        
                                        // 处理剩余元素
                                        for (; j < j_max; ++j) {
                                            C[c_base + j] += a_val * B[b_base + j];
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

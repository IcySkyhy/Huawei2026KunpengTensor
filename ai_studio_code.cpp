#include <random>
#include <sys/time.h>
#include <omp.h>
#include <cassert>
#include <cmath>
#include <arm_neon.h> // 引入 NEON 头文件

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

// 核心优化函数
void tensor_contraction(int S, int L, int M, int I, int J, const float *A, const float *B, float *C)
{
    // 计算Stride，避免重复计算
    const int strideA = L * I;
    const int strideB = M * J;
    const int strideC = M * I * J;

    // L和M层级并行，collapse(2)增加并行度
    #pragma omp parallel for collapse(2) schedule(static)
    for (int l = 0; l < L; ++l) {
        for (int m = 0; m < M; ++m) {
            
            float* current_C = C + l * strideC + m * I * J;

            // 对 I 和 J 进行分块处理
            // I 维度每次处理 2 行
            // J 维度每次处理 8 列 (适配 NEON, 8个float = 4个doublex2)
            for (int i = 0; i < I; i += 2) {
                for (int j = 0; j < J; j += 8) {
                    
                    // 初始化累加器 (Double Precision)
                    // c0_x 对应第 i 行, c1_x 对应第 i+1 行
                    // 覆盖 j 到 j+7 (8个元素)
                    float64x2_t c0_0 = vdupq_n_f64(0.0);
                    float64x2_t c0_1 = vdupq_n_f64(0.0);
                    float64x2_t c0_2 = vdupq_n_f64(0.0);
                    float64x2_t c0_3 = vdupq_n_f64(0.0);

                    float64x2_t c1_0 = vdupq_n_f64(0.0);
                    float64x2_t c1_1 = vdupq_n_f64(0.0);
                    float64x2_t c1_2 = vdupq_n_f64(0.0);
                    float64x2_t c1_3 = vdupq_n_f64(0.0);

                    const float* A_ptr = A + l * I + i;
                    const float* B_ptr = B + m * J + j;

                    for (int s = 0; s < S; ++s) {
                        // 预取下一轮数据 (Stride 很大，预取非常重要)
                        __builtin_prefetch(A_ptr + strideA, 0, 0); 
                        __builtin_prefetch(B_ptr + strideB, 0, 0);

                        // 加载 B 的 8 个 float
                        float32x4_t b_vec_lo = vld1q_f32(B_ptr);      // B[j...j+3]
                        float32x4_t b_vec_hi = vld1q_f32(B_ptr + 4);  // B[j+4...j+7]

                        // 将 B 转为 double (低位和高位)
                        float64x2_t b0 = vcvt_f64_f32(vget_low_f32(b_vec_lo));  // B[j, j+1]
                        float64x2_t b1 = vcvt_f64_f32(vget_high_f32(b_vec_lo)); // B[j+2, j+3]
                        float64x2_t b2 = vcvt_f64_f32(vget_low_f32(b_vec_hi));  // B[j+4, j+5]
                        float64x2_t b3 = vcvt_f64_f32(vget_high_f32(b_vec_hi)); // B[j+6, j+7]

                        // 处理第 i 行 (A[i])
                        {
                            double a_val = (double)(*A_ptr); // 加载 A[i] 并转为 double
                            float64x2_t a_vec = vdupq_n_f64(a_val); // 广播

                            // FMLA: c += a * b
                            c0_0 = vfmaq_f64(c0_0, b0, a_vec);
                            c0_1 = vfmaq_f64(c0_1, b1, a_vec);
                            c0_2 = vfmaq_f64(c0_2, b2, a_vec);
                            c0_3 = vfmaq_f64(c0_3, b3, a_vec);
                        }

                        // 处理第 i+1 行 (A[i+1])
                        {
                            double a_val = (double)(*(A_ptr + 1));
                            float64x2_t a_vec = vdupq_n_f64(a_val);

                            c1_0 = vfmaq_f64(c1_0, b0, a_vec);
                            c1_1 = vfmaq_f64(c1_1, b1, a_vec);
                            c1_2 = vfmaq_f64(c1_2, b2, a_vec);
                            c1_3 = vfmaq_f64(c1_3, b3, a_vec);
                        }

                        // 指针移动到下一个 S
                        A_ptr += strideA;
                        B_ptr += strideB;
                    }

                    // 将累加结果存回 C (Double -> Float)
                    // Row i
                    {
                        float32x4_t res_lo = vcombine_f32(vcvt_f32_f64(c0_0), vcvt_f32_f64(c0_1));
                        float32x4_t res_hi = vcombine_f32(vcvt_f32_f64(c0_2), vcvt_f32_f64(c0_3));
                        vst1q_f32(current_C + i * J + j, res_lo);
                        vst1q_f32(current_C + i * J + j + 4, res_hi);
                    }
                    // Row i+1
                    {
                        float32x4_t res_lo = vcombine_f32(vcvt_f32_f64(c1_0), vcvt_f32_f64(c1_1));
                        float32x4_t res_hi = vcombine_f32(vcvt_f32_f64(c1_2), vcvt_f32_f64(c1_3));
                        vst1q_f32(current_C + (i + 1) * J + j, res_lo);
                        vst1q_f32(current_C + (i + 1) * J + j + 4, res_hi);
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
        printf("PASSED\n");
    } else {
        printf("FAILED!!! max_diff = %lf\n", max_diff);
    }
}

int main()
{
    srand(time(NULL));
    // 题目规模
    tensor_contraction_test(256, 256, 256, 32, 32);
    return 0;
}
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
    #pragma omp parallel for collapse(2) schedule(static)
    for (int l = 0; l < L; ++l) {
        for (int m = 0; m < M; ++m) {
            // Local accumulator for C[l][m][I][J]
            // Using double for precision as required
            double c_local[32][32];
            
            // Initialize accumulators
            for (int i = 0; i < I; ++i) {
                for (int j = 0; j < J; ++j) {
                    c_local[i][j] = 0.0;
                }
            }
            
            for (int s = 0; s < S; ++s) {
                const float* A_ptr = A + (s * L + l) * I;
                const float* B_ptr = B + (s * M + m) * J;
                
                for (int i = 0; i < I; ++i) {
                    double a_val = (double)A_ptr[i];
                    for (int j = 0; j < J; ++j) {
                        c_local[i][j] += a_val * (double)B_ptr[j];
                    }
                }
            }
            
            // Store result
            float* C_ptr = C + (l * M + m) * I * J;
            for (int i = 0; i < I; ++i) {
                for (int j = 0; j < J; ++j) {
                    C_ptr[i * J + j] = (float)c_local[i][j];
                }
            }
        }
    }
}

#include <malloc.h>

void* my_aligned_alloc(size_t alignment, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    return aligned_alloc(alignment, size);
#endif
}

void my_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

void tensor_contraction_test(int S, int L, int M, int I, int J)
{
    static constexpr int test_times = 10;
    float *A = (float *)my_aligned_alloc(64, S * L * I * test_times * sizeof(float));
    float *B = (float *)my_aligned_alloc(64, S * M * J * test_times * sizeof(float));
    float *C = (float *)my_aligned_alloc(64, L * M * I * J * test_times * sizeof(float));
    float *C_std = (float *)my_aligned_alloc(64, L * M * I * J * sizeof(float));

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

    my_free(A);
    my_free(B);
    my_free(C);
    my_free(C_std);

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
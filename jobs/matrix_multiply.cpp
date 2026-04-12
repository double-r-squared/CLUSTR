#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>

// Simple NxN matrix multiply — first HPC test job for CLUSTR.
// Compile:  g++ -O2 -o matrix_multiply matrix_multiply.cpp
// Run:      ./matrix_multiply

int main() {
    const int N = 256;

    std::cout << "CLUSTR test job: matrix multiply " << N << "x" << N << std::endl;

    std::vector<std::vector<double>> A(N, std::vector<double>(N));
    std::vector<std::vector<double>> B(N, std::vector<double>(N));
    std::vector<std::vector<double>> C(N, std::vector<double>(N, 0.0));

    // Deterministic seed so result is verifiable
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = dist(rng);
            B[i][j] = dist(rng);
        }

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
                C[i][j] += A[i][k] * B[k][j];

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    double checksum = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            checksum += C[i][j];

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Checksum: " << checksum  << std::endl;
    std::cout << "Time:     " << elapsed   << "s" << std::endl;
    std::cout << "GFLOPS:   "
              << (2.0 * N * N * N / elapsed / 1e9) << std::endl;

    return 0;
}

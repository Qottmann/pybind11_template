#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <Accelerate/Accelerate.h>

namespace py = pybind11;
using namespace std;

double add(double x, double y)
{
    return x + y;
}

vector<vector<double>> mmul(const vector<vector<double>>& a, const vector<vector<double>>& b) {
    // Check if matrices can be multiplied
    int rows_a = a.size();
    int cols_a = a[0].size();
    int rows_b = b.size();
    int cols_b = b[0].size();

    if (cols_a != rows_b) {
        std::cerr << "Matrix dimensions are not compatible for multiplication." << std::endl;
        return std::vector<std::vector<double>>();
    }

    // Initialize the result matrix c with zeros
    vector<vector<double>> c(rows_a, vector<double>(cols_b, 0));

    // Perform matrix multiplication
    for (int i = 0; i < rows_a; ++i) {
        for (int j = 0; j < cols_b; ++j) {
            for (int k = 0; k < cols_a; ++k) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return c;
}



// Function to perform matrix multiplication using BLAS
std::vector<std::vector<double>> mmul_blas(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    int m = A.size();    // Number of rows in A
    int k = A[0].size(); // Number of columns in A (should be equal to rows in B)
    int n = B[0].size(); // Number of columns in B

    std::vector<std::vector<double>> C(m, std::vector<double>(n, 0.0));

    // Convert input matrices to flat arrays
    std::vector<double> flatA;
    for (const auto& row : A) {
        flatA.insert(flatA.end(), row.begin(), row.end());
    }

    std::vector<double> flatB;
    for (const auto& row : B) {
        flatB.insert(flatB.end(), row.begin(), row.end());
    }

    // Perform matrix multiplication: C = A * B
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, flatA.data(), k, flatB.data(), n, 0.0, C[0].data(), n);

    return C;
}



PYBIND11_MODULE(my_cpp_module, handle) {
    handle.doc() = "my cpp-python module docs";
    handle.def("add", &add);
    handle.def("mmul", &mmul);
    handle.def("mmul_blas", &mmul_blas);
}

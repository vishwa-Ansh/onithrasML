#include <pybind11/pybind11.h>

namespace py = pybind11;

double predict(double x, double weight, double bias)
{
    return weight * x + bias;
}

PYBIND11_MODULE(_backend, m)
{
    m.doc() = "Veltrix C++ backend";

    m.def(
        "predict",
        &predict,
        "Linear regression prediction"
    );
}
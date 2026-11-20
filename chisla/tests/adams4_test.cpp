#include "adams4.hpp"
#include <cassert>
#include <cmath>
double rhs(double x, double y) { return x*x*x - 2*y/x; }
double constant(double, double) { return 2; }
int main() {
    assert(std::abs(adams4(constant, 0, 1, 1, 10) - 3) < 1e-12);
    // Exact solution: y = x^4/6 + 95/(384*x^2).
    assert(std::abs(adams4(rhs, 0.5, 1, 2, 150) - (16.0/6 + 95.0/1536)) < 1e-5);
}

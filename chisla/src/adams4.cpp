#include "adams4.hpp"
#include <cmath>
#include <cstdio>
#include <limits>

double adams4(double (*f)(double, double), double start, double y0, double end, int steps) {
    if (!f || steps < 1 || !std::isfinite(start) || !std::isfinite(end) || !std::isfinite(y0) || end <= start) {
        printf("Invalid initial data\n");
        return std::numeric_limits<double>::quiet_NaN();
    }
    double h = (end - start) / steps;
    double x = start, y = y0;
    double values[4] = {}; // Last four derivative values, newest first.
    values[0] = f(x, y);
    printf("x\ty\n%.6f\t%.9f\n", x, y);

    for (int i = 1; i <= steps; i++) {
        if (i <= 3) {
            // The first three steps use Runge-Kutta of order 4.
            double k1 = f(x, y);
            double k2 = f(x + h/2, y + h*k1/2);
            double k3 = f(x + h/2, y + h*k2/2);
            double k4 = f(x + h, y + h*k3);
            y += h*(k1 + 2*k2 + 2*k3 + k4)/6;
        } else {
            y += h*(55*values[0] - 59*values[1] + 37*values[2] - 9*values[3])/24;
        }
        x = start + i*h;
        for (int j = 3; j > 0; j--) values[j] = values[j-1];
        values[0] = f(x, y);
        printf("%.6f\t%.9f\n", x, y);
    }
    return y;
}

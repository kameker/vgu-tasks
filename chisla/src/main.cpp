#include <stdio.h>
#include <cmath>
#include "methods_solutions.hpp"
#include "cube_spline.hpp"
#include "adams4.hpp"
#include "fun.hpp"

double adams4f(double x, double y){
    return pow(x, 3) - 2 * y / x;
}
double y05 = 1;
double a = 0.5;
double b = 2;

double spline[9] = {0, 2, 4, 6, -1, -4, 2, -8, 0.5};


int main() {
    adams4(adams4f, a, y05, b, 15);
    return 0;
}


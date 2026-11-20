#include "cube_spline.hpp"
#include <cassert>
#include <cmath>
int main() {
    double x[] = {0,2,4,6}, y[] = {-1,-4,2,-8};
    double a[4], b[4], c[4], d[4];
    spline_coefficients(x,y,a,b,c,d);
    for (int i=0; i<4; ++i)
        assert(std::abs(spline_value(x,a,b,c,d,x[i])-y[i])<1e-10);
    assert(std::abs(spline_value(x,a,b,c,d,0.5)+2.5625)<1e-10);
    assert(std::abs(c[3])<1e-10);
    assert(std::abs(c[1]-d[1]*(x[1]-x[0]))<1e-10);
}

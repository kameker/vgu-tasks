#ifndef CUBE_SPLINE_HPP
#define CUBE_SPLINE_HPP

// Four nodes: x[0..3], y[0..3]. Coefficients belong to the right end of each interval.
void spline_coefficients(const double x[], const double y[], double a[], double b[], double c[], double d[]);
double spline_value(const double x[], const double a[], const double b[], const double c[], const double d[], double point);
void solve_spline(const double data[]);

#endif

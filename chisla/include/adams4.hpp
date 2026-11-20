#ifndef ADAMS4_HPP
#define ADAMS4_HPP

// Solves y'=f(x,y), prints the nodes and returns the value at the end.
double adams4(double (*f)(double, double), double start, double y0, double end, int steps);

#endif

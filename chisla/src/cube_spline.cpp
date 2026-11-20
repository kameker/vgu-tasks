#include "cube_spline.hpp"
#include <cstdio>
#include <cmath>

// Natural spline: c[0] = c[3] = 0.
void spline_coefficients(const double x[], const double y[], double a[], double b[], double c[], double d[]) {
    double h[4], alpha[4] = {}, beta[4] = {};
    for (int i = 1; i <= 3; i++) h[i] = x[i] - x[i-1];

    // Thomas algorithm for c[1] and c[2].
    for (int i = 1; i <= 2; i++) {
        double right = 6 * ((y[i+1]-y[i])/h[i+1] - (y[i]-y[i-1])/h[i]);
        double denominator = 2*(h[i]+h[i+1]) + h[i]*alpha[i-1];
        alpha[i] = -h[i+1]/denominator;
        beta[i] = (right-h[i]*beta[i-1])/denominator;
    }
    c[0] = c[3] = 0;
    for (int i = 2; i >= 1; i--) c[i] = alpha[i]*c[i+1] + beta[i];
    for (int i = 1; i <= 3; i++) {
        a[i] = y[i];
        d[i] = (c[i]-c[i-1])/h[i];
        b[i] = (y[i]-y[i-1])/h[i] + h[i]*c[i]/2 - h[i]*h[i]*d[i]/6;
    }
}

double spline_value(const double x[], const double a[], const double b[], const double c[], const double d[], double point) {
    int i = 1;
    while (i < 3 && point > x[i]) i++;
    double t = point - x[i];
    return a[i] + b[i]*t + c[i]*t*t/2 + d[i]*t*t*t/6;
}

void save_spline_plot(const double x[], const double y[], const double a[], const double b[], const double c[], const double d[]) {
    double low = y[0], high = y[0];
    for (int i = 0; i <= 400; i++) {
        double point = x[0] + (x[3]-x[0])*i/400;
        double value = spline_value(x,a,b,c,d,point);
        if (value < low) low = value;
        if (value > high) high = value;
    }
    if (low == high) { low--; high++; }
    double margin = (high-low)*0.1;
    low -= margin; high += margin;
    FILE* file = fopen("spline.svg", "w");
    if (!file) { printf("Cannot create spline.svg\n"); return; }
    fprintf(file, "<svg xmlns='http://www.w3.org/2000/svg' width='800' height='500'>\n");
    fprintf(file, "<rect width='800' height='500' fill='white'/><text x='60' y='25'>Natural cubic spline</text><path d='M60 40 V440 H760' fill='none' stroke='black'/><polyline fill='none' stroke='blue' stroke-width='2' points='");
    for (int i = 0; i <= 400; i++) {
        double point = x[0] + (x[3]-x[0])*i/400;
        double value = spline_value(x,a,b,c,d,point);
        fprintf(file, "%g,%g ", 60+700.0*i/400, 440-400*(value-low)/(high-low));
    }
    fprintf(file, "'/>\n");
    for (int i = 0; i < 4; i++) {
        double px = 60+700*(x[i]-x[0])/(x[3]-x[0]);
        double py = 440-400*(y[i]-low)/(high-low);
        fprintf(file, "<circle cx='%g' cy='%g' r='5' fill='red'/><text x='%g' y='%g'>(%g, %g)</text>\n", px, py, px+7, py-7, x[i], y[i]);
    }
    fprintf(file, "</svg>\n");
    if (fclose(file) != 0) printf("Cannot finish writing spline.svg\n");
    else printf("Graph saved: spline.svg\n");
}

void solve_spline(const double data[]) {
    const double* x = data;
    const double* y = data + 4;
    double point = data[8];
    double a[4] = {}, b[4] = {}, c[4] = {}, d[4] = {};
    for (int i = 0; i < 9; i++) {
        if (!std::isfinite(data[i])) { printf("All data must be finite\n"); return; }
    }
    for (int i = 1; i <= 3; i++) {
        if (x[i] <= x[i-1]) { printf("Nodes must increase\n"); return; }
    }
    if (point < x[0] || point > x[3]) { printf("Point is outside the interpolation interval\n"); return; }
    spline_coefficients(x,y,a,b,c,d);
    printf("S(x) = a + b*(x-xi) + c/2*(x-xi)^2 + d/6*(x-xi)^3\n");
    for (int i = 1; i <= 3; i++) {
        printf("[%g, %g]: a=%.6f b=%.6f c=%.6f d=%.6f\n", x[i-1], x[i], a[i], b[i], c[i], d[i]);
        printf("S%d(x) = %.6f + (%.6f)*(x-%g) + (%.6f)*(x-%g)^2 + (%.6f)*(x-%g)^3\n", i, a[i], b[i], x[i], c[i]/2, x[i], d[i]/6, x[i]);
    }
    printf("\nNode check:\n");
    for (int i = 0; i < 4; i++)
        printf("x=%g y=%g S(x)=%.6f\n", x[i], y[i], spline_value(x,a,b,c,d,x[i]));
    printf("\nS(%g) = %.6f\n", point, spline_value(x,a,b,c,d,point));
    save_spline_plot(x,y,a,b,c,d);
}



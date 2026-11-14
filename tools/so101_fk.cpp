#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

// Topologically ordered frames: parent, home transform (row-major), local axis, q.
// A root has parent -1. All distances are metres and angles are radians.
using Mat = std::array<double, 16>;
Mat mul(const Mat &a, const Mat &b) {
    Mat c{};
    for (int i=0;i<4;++i) for (int j=0;j<4;++j)
        for (int k=0;k<4;++k) c[i*4+j] += a[i*4+k]*b[k*4+j];
    return c;
}
int main() {
    int n;
    if (!(std::cin >> n) || n < 1) return 1;
    std::vector<Mat> world;
    std::cout << std::setprecision(17);
    for (int i=0;i<n;++i) {
        int parent; Mat h{}; double x,y,z,q;
        std::cin >> parent;
        for (auto &v:h) std::cin >> v;
        std::cin >> x >> y >> z >> q;
        if (!std::cin || parent >= i || parent < -1) return 2;
        double norm=std::sqrt(x*x+y*y+z*z);
        if (norm==0) return 3;
        x/=norm; y/=norm; z/=norm;
        double c=std::cos(q), s=std::sin(q), t=1-c;
        Mat r={t*x*x+c,t*x*y-s*z,t*x*z+s*y,0,
               t*x*y+s*z,t*y*y+c,t*y*z-s*x,0,
               t*x*z-s*y,t*y*z+s*x,t*z*z+c,0,0,0,0,1};
        Mat m=mul(h,r);
        if (parent>=0) m=mul(world[parent],m);
        world.push_back(m);
        for (double v:m) std::cout << v << ' ';
        std::cout << '\n';
    }
}

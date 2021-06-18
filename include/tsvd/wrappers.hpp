#ifndef TSVD_EIGEN_WRAPPERS_HPP
#define TSVD_EIGEN_WRAPPERS_HPP

namespace tsvd {

template<class EIGEN>
class EigenWrapper {
public:
    EigenWrapper(EIGEN m) : mat(std::move(m)) {}

    template<class OTHER>
    friend auto operator*(const EigenWrapper<EIGEN>& x, const OTHER& y) {
        return x.mat * y;
    }

    template<class OTHER>
    friend auto operator*(const OTHER& x, const EigenWrapper<EIGEN>& y) {
        return x * y.mat;
    }

    template<class OTHER>
    friend auto crossprod(const EigenWrapper<EIGEN>& x, const OTHER& y) {
        return x.mat.cross(y);
    }

    template<class OTHER>
    friend auto crossprod(const OTHER& x, const EigenWrapper<EIGEN>& x) {
        return x.cross(y.mat);
    }
private:
    EIGEN mat;
};

}

#endif

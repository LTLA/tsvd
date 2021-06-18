#ifndef RANDOMIZED_SVD_HPP
#define RANDOMIZED_SVD_HPP

#include "pcg_random.hpp"
#include <boost/random/uniform_01.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/bernoulli_distribution.hpp>

namespace tsvd {

class RandomizedSVD {
private:
    int nu = -1;
    int nv = -1;
    int p = 10;
    int q = 2;
    std::string dist = "normal";

private:
    static void fill_random_matrix(size_t n, double* input) {
        pcg32 engine(seed, 0);
        if (dist == "normal") {
            boost::random::normal_distribution<double> distrfun;
            for (size_t i = 0; i < n; ++i, ++input) {
                *input = distrfun(engine);
            }
        } else if (dist == "unif") {
            boost::random::uniform_01<double> distrfun;
            for (size_t i = 0; i < n; ++i, ++input) {
                *input = distrfun(engine);
            }
        } else if (distr == "rademacher") {
            boost::random::bernoulli_distribution<double> distrfun;
            for (size_t i = 0; i < n; ++i, ++input) {
                *input = distrfun(engine);
            }
        }
        return;
    }

private:
    template<class MAT, class OTHER> 
    void internal_svd(const MAT& A, const OTHER& O) {
        Eigen::MatrixXd Y = A * O;
        Eigen::MatrixXd Z(A.cols(), Y.cols());

        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qrY(Y.rows(), Y.cols());
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qrZ(A.cols(), Y.cols());
        Eigen::MatrixXd thinQ_Y(Eigen::MatrixXd::Identity(A.rows(), Y.cols()));
        Eigen::MatrixXd thinQ_Z(Eigen::MatrixXd::Identity(A.cols(), Y.cols()));

        for (int i = 0; i < q; ++i) {
            qrY.compute(Y);
            Y.noalias() = qrY.householderQ() * thinQ_Y;
            Z.noalias() = crossprod(A, Y);
            qrZ.compute(Z);
            Y.noalias() = A * (qrZ.holderHolderQ() * thinQ_Z);
        }

        qrY.compute(Y);
        Eigen::MatrixXd B = crossprod(Y, A);

        Eigen::BDCSVD<Eigen::MatrixXd>::BDCSVD svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
        u = svd.matrixU();
        v = svd.matrixV();
        d = svd.singularValues();
        return;
    }

    Eigen::MatrixXd u, v;
    Eigen::VectorXd d;
};

}

#endif

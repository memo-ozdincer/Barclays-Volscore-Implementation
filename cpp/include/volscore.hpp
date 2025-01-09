#ifndef VOL_SCORE_HPP
#define VOL_SCORE_HPP

#include <vector>

/**
 * VolScore class
 * 
 * Computes realized volatility metrics on a sequence of price data, 
 * and extends to measure higher-order statistics (skew/kurt) that might be used 
 * in more advanced volatility modeling. This is only illustrative code.
 */
class VolScore {
public:
    // Compute standard realized volatility of daily returns
    double computeRealizedVol(const std::vector<double>& prices);

    // Return the 'VolScore' (multiplied realized vol) for simple comparisons
    double computeVolScore(const std::vector<double>& prices);

    // Compute realized skewness of returns
    double computeRealizedSkew(const std::vector<double>& prices);

    // Compute realized excess kurtosis
    double computeRealizedKurt(const std::vector<double>& prices);

private:
    double computeStdDev(const std::vector<double>& returns);
    double computeMean(const std::vector<double>& data);
};

#endif
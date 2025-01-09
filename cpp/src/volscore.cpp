#include "volscore.hpp"
#include <numeric>
#include <cmath>
#include <stdexcept>

double VolScore::computeMean(const std::vector<double>& data) {
    if(data.empty()) {
        throw std::runtime_error("Data vector is empty, cannot compute mean.");
    }
    double sum = 0.0;
    for(auto val : data) {
        sum += val;
    }
    return sum / static_cast<double>(data.size());
}

double VolScore::computeStdDev(const std::vector<double>& returns) {
    double mean = computeMean(returns);
    double var = 0.0;
    for(auto r : returns) {
        double diff = (r - mean);
        var += diff * diff;
    }
    var /= static_cast<double>(returns.size());
    return std::sqrt(var);
}

double VolScore::computeRealizedVol(const std::vector<double>& prices) {
    if(prices.size() < 2) {
        throw std::runtime_error("Not enough price points for realized vol");
    }
    std::vector<double> rets;
    rets.reserve(prices.size() - 1);
    for(size_t i = 1; i < prices.size(); ++i) {
        double dailyRet = (prices[i] - prices[i-1]) / prices[i-1];
        rets.push_back(dailyRet);
    }
    return computeStdDev(rets);
}

double VolScore::computeVolScore(const std::vector<double>& prices) {
    double rv = computeRealizedVol(prices);
    // Just scale realized vol by 100 for a "VolScore"
    return 100.0 * rv;
}

double VolScore::computeRealizedSkew(const std::vector<double>& prices) {
    if(prices.size() < 3) {
        return 0.0;
    }
    // Convert to returns
    std::vector<double> rets;
    rets.reserve(prices.size() - 1);
    for(size_t i = 1; i < prices.size(); ++i) {
        double dailyRet = (prices[i] - prices[i-1]) / prices[i-1];
        rets.push_back(dailyRet);
    }
    double mean = computeMean(rets);
    double m3 = 0.0;
    double m2 = 0.0;
    for(auto r : rets) {
        double diff = r - mean;
        m3 += diff * diff * diff;
        m2 += diff * diff;
    }
    m3 /= rets.size();
    m2 /= rets.size();
    double stdev = std::sqrt(m2);
    if(std::fabs(stdev) < 1e-12) {
        return 0.0;
    }
    return m3 / (stdev * stdev * stdev);
}

double VolScore::computeRealizedKurt(const std::vector<double>& prices) {
    if(prices.size() < 3) {
        return 0.0;
    }
    // Convert to returns
    std::vector<double> rets;
    rets.reserve(prices.size() - 1);
    for(size_t i = 1; i < prices.size(); ++i) {
        double dailyRet = (prices[i] - prices[i-1]) / prices[i-1];
        rets.push_back(dailyRet);
    }
    double mean = computeMean(rets);
    double m4 = 0.0;
    double m2 = 0.0;
    for(auto r : rets) {
        double diff = r - mean;
        m4 += diff * diff * diff * diff;
        m2 += diff * diff;
    }
    m4 /= rets.size();
    m2 /= rets.size();
    double stdev = std::sqrt(m2);
    if(std::fabs(stdev) < 1e-12) {
        return 0.0;
    }
    // Excess kurtosis = M4 / (StdDev^4) - 3
    return (m4 / (stdev * stdev * stdev * stdev)) - 3.0;
}
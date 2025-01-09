#include <iostream>
#include <vector>
#include "../include/volscore.hpp"

int main() {
    VolScore vs;
    std::vector<double> mockData = {100, 101, 105, 102, 98, 99};
    
    try {
        double rv = vs.computeRealizedVol(mockData);
        double vs_val = vs.computeVolScore(mockData);
        std::cout << "Realized Vol: " << rv << std::endl;
        std::cout << "VolScore: " << vs_val << std::endl;
        return 0; // success
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}

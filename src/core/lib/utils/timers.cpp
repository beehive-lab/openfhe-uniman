/**
 * Utility for measuring time spent in functions of GPU Acceleration.
 * This utility reuses and based on timer infrastructure of OpenFHE (see debug.h)
 *
 * Note: to activate use -DCMAKE_BUILD_TYPE=Debug option in cmake.
 */

#include "utils/timers.h"

#include <iostream>

namespace lbcrypto {

int numOfHomomorphicOperations = 0;

double applicationTimer = 0;
double operationsTimer = 0;

double approxModDownTimer_CPU = 0;
double approxModDownTimer_GPU = 0;

void setNumOfOperations(int numOfOperations) {
    numOfHomomorphicOperations = numOfOperations;
}

void accumulateTimer(double &timer, double toc) {
    timer += toc;
}

void printTimers() {
    std::cout << "Total execution time = "                          << applicationTimer << " ms" << std::endl;
    std::cout << "Homomorphic Operations Time = "                   << operationsTimer << " ms" << std::endl;
    std::cout << "Homomorphic Operation/sec = "                     << (numOfHomomorphicOperations * 1000) / operationsTimer  << std::endl;
    std::cout << "====================================="            << std::endl;
    std::cout << "Total time in ApproxModDown_CPU = "               << approxModDownTimer_CPU << " ms" << std::endl;
    std::cout << "Total time in ApproxModDown_GPU = "               << approxModDownTimer_GPU << " ms" << std::endl;
}

};
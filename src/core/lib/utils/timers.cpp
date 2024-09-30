/**
 * Utility for measuring time spent in functions of GPU Acceleration.
 * This utility reuses and based on timer infrastructure of OpenFHE (see debug.h)
 *
 * Note: to activate use -DCMAKE_BUILD_TYPE=Debug option in cmake.
 */

#include "utils/timers.h"

#include <iostream>

namespace lbcrypto {

double evalKeySwitchPrecomputeCoreTimer_CPU = 0;
double evalKeySwitchPrecomputeCoreTimer_GPU = 0;
double evalFastKeySwitchCoreTimer_CPU = 0;
double evalFastKeySwitchCoreTimer_GPU = 0;

void accumulateTimer(double &timer, double toc) {
    timer += toc;
}

double getEvalKeySwitchPrecomputeCoreTimer_CPU() {
    return evalKeySwitchPrecomputeCoreTimer_CPU;
}

double getEvalKeySwitchPrecomputeCoreTimer_GPU() {
    return evalKeySwitchPrecomputeCoreTimer_GPU;
}

double getEvalFastKeySwitchCoreTimer_CPU() {
    return evalFastKeySwitchCoreTimer_CPU;
}

double getEvalFastKeySwitchCoreTimer_GPU() {
    return evalFastKeySwitchCoreTimer_GPU;
}

void printTimers() {
    std::cout << "Total time in evalKeySwitchPrecomputeCoreTimer_CPU = " << getEvalKeySwitchPrecomputeCoreTimer_CPU() << "ms" << std::endl;
    std::cout << "Total time in evalKeySwitchPrecomputeCoreTimer_GPU = " << getEvalKeySwitchPrecomputeCoreTimer_GPU() << "ms" << std::endl;
    std::cout << "Total time in evalFastKeySwitchCoreTimer_CPU = " << getEvalFastKeySwitchCoreTimer_CPU() << "ms" << std::endl;
    std::cout << "Total time in evalFastKeySwitchCoreTimer_GPU = " << getEvalFastKeySwitchCoreTimer_GPU() << "ms" << std::endl;
}

};
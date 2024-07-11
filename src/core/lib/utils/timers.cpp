/**
 * Utility for measuring time spent in functions of GPU Acceleration.
 * This utility reuses and based on timer infrastructure of OpenFHE (see debug.h)
 *
 * Note: to activate use -DCMAKE_BUILD_TYPE=Debug option in cmake.
 */

#include "utils/timers.h"

#include <iostream>

namespace lbcrypto {

double approxSwitchTimer = 0;

void accumulateTimer(double &timer, double toc) {
    timer += toc;
}

double getApproxSwitchTimer() {
    return approxSwitchTimer;
}

void printTimers() {
    std::cout << "Total time in ApproxSwitchCRTBasis = " << getApproxSwitchTimer() << "ms" << std::endl;
}

};
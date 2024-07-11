/**
* Utility for measuring time spent in functions of GPU Acceleration.
 * This utility reuses and based on timer infrastructure of OpenFHE (see debug.h)
 *
 * Note: to activate use -DCMAKE_BUILD_TYPE=Debug option in cmake.
 */

namespace lbcrypto {

extern double approxSwitchTimer;

void accumulateTimer(double &timer, double toc);

double getApproxSwitchTimer();

void printTimers();

}


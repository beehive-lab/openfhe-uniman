/**
 * Utility for measuring time spent in functions of GPU Acceleration.
 * This utility reuses and based on timer infrastructure of OpenFHE (see debug.h)
 *
 * Note: to activate use -DCMAKE_BUILD_TYPE=Debug option in cmake.
 */

#include "utils/timers.h"

#include <iostream>

namespace lbcrypto {

double approxSwitchTimer_CPU = 0;
double approxSwitchTimer_GPU = 0;
double approxModDownTimer_CPU = 0;
double approxModDownTimer_GPU = 0;

void accumulateTimer(double &timer, double toc) {
    timer += toc;
}

double getApproxSwitchTimerCPU() {
    return approxSwitchTimer_CPU;
}

double getApproxSwitchTimerGPU() {
    return approxSwitchTimer_GPU;
}

double getApproxModDownTimerCPU() {
    return approxSwitchTimer_CPU;
}

double getApproxModDownTimerGPU() {
    return approxModDownTimer_GPU;
}

void printTimers() {
    std::cout << "Total time in ApproxSwitchCRTBasis_CPU = " << getApproxSwitchTimerCPU() << "ms" << std::endl;
    std::cout << "Total time in ApproxSwitchCRTBasis_GPU = " << getApproxSwitchTimerGPU() << "ms" << std::endl;
    std::cout << "Total time in ApproxModDown_CPU = " << getApproxModDownTimerCPU() << "ms" << std::endl;
    std::cout << "Total time in ApproxModDown_GPU = " << getApproxModDownTimerGPU() << "ms" << std::endl;
}

};
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

int    approxSwitchCRTBasisCounter_CPU = 0;
int    approxSwitchCRTBasisCounter_GPU = 0;
int    approxModDownCounter_CPU = 0;
int    approxModDownCounter_GPU = 0;

void accumulateTimer(double &timer, double toc) {
    timer += toc;
}

void incrementInvocationCounter(int &counter) {
    counter++;
}

double getApproxSwitchTimerCPU() {
    return approxSwitchTimer_CPU;
}

double getApproxSwitchTimerGPU() {
    return approxSwitchTimer_GPU;
}

double getApproxModDownTimerCPU() {
    return approxModDownTimer_CPU;
}

double getApproxModDownTimerGPU() {
    return approxModDownTimer_GPU;
}

int getApproxSwitchCRTBasisCounter_CPU() { return approxSwitchCRTBasisCounter_CPU; }
int getApproxSwitchCRTBasisCounter_GPU() { return approxSwitchCRTBasisCounter_GPU; }
int getApproxModDownCounter_CPU() { return approxModDownCounter_CPU; }
int getApproxModDownCounter_GPU() { return approxModDownCounter_GPU; }

void printTimers() {
    std::cout << "Total time in ApproxSwitchCRTBasis_CPU = " << getApproxSwitchTimerCPU() << "ms ( " << getApproxSwitchCRTBasisCounter_CPU() << " invocations) " << std::endl;
    std::cout << "Total time in ApproxSwitchCRTBasis_GPU = " << getApproxSwitchTimerGPU() << "ms ( " << getApproxSwitchCRTBasisCounter_GPU() << " invocations) " << std::endl;
    std::cout << "Total time in ApproxModDown_CPU = " << getApproxModDownTimerCPU() << "ms ( " << getApproxModDownCounter_CPU() << " invocations) " << std::endl;
    std::cout << "Total time in ApproxModDown_GPU = " << getApproxModDownTimerGPU() << "ms ( " << getApproxModDownCounter_GPU() << " invocations) " << std::endl;
}

};
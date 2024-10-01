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

int    evalKeySwitchPrecomputeCoreCounter_CPU = 0;
int    evalKeySwitchPrecomputeCoreCounter_GPU = 0;
int    evalFastKeySwitchCoreCounter_CPU = 0;
int    evalFastKeySwitchCoreCounter_GPU = 0;

void accumulateTimer(double &timer, double toc) {
    timer += toc;
}

void incrementInvocationCounter(int &counter) {
    counter++;
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

/*int getApproxSwitchCRTBasisCounter_CPU() { return approxSwitchCRTBasisCounter_CPU; }
int getApproxSwitchCRTBasisCounter_GPU() { return approxSwitchCRTBasisCounter_GPU; }
int getApproxModDownCounter_CPU() { return approxModDownCounter_CPU; }
int getApproxModDownCounter_GPU() { return approxModDownCounter_GPU; }*/

void printTimers() {
    std::cout << "Total time in EvalKeySwitchPrecomputeCore_CPU = " << getEvalKeySwitchPrecomputeCoreTimer_CPU() << "ms " << std::endl;
    std::cout << "Total time in EvalKeySwitchPrecomputeCore_GPU = " << getEvalKeySwitchPrecomputeCoreTimer_GPU() << "ms " << std::endl;
    std::cout << "Total time in EvalFastKeySwitchCore_CPU = " << getEvalFastKeySwitchCoreTimer_CPU() << "ms ( " << std::endl;
    std::cout << "Total time in EvalFastKeySwitchCore_GPU = " << getEvalFastKeySwitchCoreTimer_GPU() << "ms ( " << std::endl;
}

};
/**
 * Utility for measuring time spent in functions of GPU Acceleration.
 * This utility reuses and based on timer infrastructure of OpenFHE (see debug.h)
 *
 * Note: to activate use -DCMAKE_BUILD_TYPE=Debug option in cmake.
 */

#include "utils/timers.h"

#include <iostream>

namespace lbcrypto {

double application = 0;

double evalKeySwitchPrecomputeCoreTimer_CPU = 0;
double evalKeySwitchPrecomputeCoreTimer_GPU = 0;
double evalFastKeySwitchCoreTimer_CPU = 0;
double evalFastKeySwitchCoreTimer_GPU = 0;

double approxModDown_total = 0;
double approxModDown_pre = 0;
double approxModDown_post = 0;

int approxModDown_invocations = 0;

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
    std::cout << "Total execution time = " << application << " ms" << std::endl;
    std::cout << "Total time in EvalKeySwitchPrecomputeCore_CPU = " << getEvalKeySwitchPrecomputeCoreTimer_CPU() << " ms" << std::endl;
    std::cout << "Total time in EvalKeySwitchPrecomputeCore_GPU = " << getEvalKeySwitchPrecomputeCoreTimer_GPU() << " ms" << std::endl;
    std::cout << "Total time in EvalFastKeySwitchCore_CPU = " << getEvalFastKeySwitchCoreTimer_CPU() << " ms" << std::endl;
    std::cout << "Total time in EvalFastKeySwitchCore_GPU = " << getEvalFastKeySwitchCoreTimer_GPU() << " ms" << std::endl;
    std::cout << "ApproxModDown: Total = " << approxModDown_total << " ms\n\tApproxModDown pre-proc = " << approxModDown_pre << " ms\n\tApproxModDown post-proc = " << approxModDown_post << " ms\n\tApproxModDown invocations = " << approxModDown_invocations << std::endl;
}

};
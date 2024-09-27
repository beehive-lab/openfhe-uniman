/**
* Utility for measuring time spent in functions of GPU Acceleration.
 * This utility reuses and based on timer infrastructure of OpenFHE (see debug.h)
 *
 * Note: to activate use -DCMAKE_BUILD_TYPE=Debug option in cmake.
 */

namespace lbcrypto {

extern double approxSwitchTimer_CPU;
extern double approxSwitchTimer_GPU;
extern double approxModDownTimer_CPU;
extern double approxModDownTimer_GPU;

void accumulateTimer(double &timer, double toc);

double getApproxSwitchTimerCPU();
double getApproxSwitchTimerGPU();
double getApproxModDownTimerCPU();
double getApproxModDownTimerGPU();

void printTimers();

}


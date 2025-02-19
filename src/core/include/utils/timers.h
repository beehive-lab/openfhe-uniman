/**
* Utility for measuring time spent in functions of GPU Acceleration.
 * This utility reuses and based on timer infrastructure of OpenFHE (see debug.h)
 *
 * Note: to activate use -DCMAKE_BUILD_TYPE=Debug option in cmake.
 */

namespace lbcrypto {

extern double applicationTimer;
extern double operationsTimer;

extern double approxModDownTimer_CPU;
extern double approxModDownTimer_GPU;

void setNumOfOperations(int numOfOperations);
void accumulateTimer(double &timer, double toc);

void printTimers();

}


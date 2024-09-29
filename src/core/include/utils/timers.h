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

extern int    approxSwitchCRTBasisCounter_CPU;
extern int    approxSwitchCRTBasisCounter_GPU;
extern int    approxModDownCounter_CPU;
extern int    approxModDownCounter_GPU;

void accumulateTimer(double &timer, double toc);
void incrementInvocationCounter(int &counter);

double getApproxSwitchTimerCPU();
double getApproxSwitchTimerGPU();
double getApproxModDownTimerCPU();
double getApproxModDownTimerGPU();

int getApproxSwitchCRTBasisCounter_CPU();
int getApproxSwitchCRTBasisCounter_GPU();
int getApproxModDownCounter_CPU();
int getApproxModDownCounter_GPU();

void printTimers();

}


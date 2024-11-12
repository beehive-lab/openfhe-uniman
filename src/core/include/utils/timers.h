/**
* Utility for measuring time spent in functions of GPU Acceleration.
 * This utility reuses and based on timer infrastructure of OpenFHE (see debug.h)
 *
 * Note: to activate use -DCMAKE_BUILD_TYPE=Debug option in cmake.
 */

namespace lbcrypto {

extern double application;

extern double evalKeySwitchPrecomputeCoreTimer_CPU;
extern double evalKeySwitchPrecomputeCoreTimer_GPU;
extern double evalFastKeySwitchCoreTimer_CPU;
extern double evalFastKeySwitchCoreTimer_GPU;

extern double approxModDown_total;
extern double approxModDown_pre;
extern double approxModDown_post;

extern double switchFormatTimerCPU;
extern double switchFormatTimerGPU;

extern int approxModDown_invocations;

/*extern int    evalKeySwitchPrecomputeCoreCounter_CPU;
extern int    evalKeySwitchPrecomputeCoreCounter_GPU;
extern int    evalFastKeySwitchCoreCounter_CPU;
extern int    evalFastKeySwitchCoreCounter_GPU;*/

void accumulateTimer(double &timer, double toc);
void incrementInvocationCounter(int &counter);

double getEvalKeySwitchPrecomputeCoreTimer_CPU();
double getEvalKeySwitchPrecomputeCoreTimer_GPU();
double getEvalFastKeySwitchCoreTimer_CPU();
double getEvalFastKeySwitchCoreTimer_GPU();

/*int getApproxSwitchCRTBasisCounter_CPU();
int getApproxSwitchCRTBasisCounter_GPU();
int getApproxModDownCounter_CPU();
int getApproxModDownCounter_GPU();*/

void printTimers();

}


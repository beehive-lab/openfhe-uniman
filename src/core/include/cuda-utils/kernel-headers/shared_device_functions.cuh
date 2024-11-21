#ifndef SHARED_DEVICE_FUNCTIONS_CUH
#define SHARED_DEVICE_FUNCTIONS_CUH

#include <cstdint> // for uint32_t type
#include <iostream> // for printf
//#include <cuda.h>
#include <cuda_runtime.h>
//#include <device_functions.h>

using uint128_t = unsigned __int128;

__device__ inline ulong ModSubFast(ulong a, ulong b, ulong modulus) {
    return a >= b ? a - b : a + (modulus - b);
}

/**
 * CUDA implementation of:
 * NativeIntegerT ModMulFastConst(const NativeIntegerT& b, const NativeIntegerT& modulus, const NativeIntegerT& bInv)
 * from core/include/math/hal/intnat/ubintnat.h
 *
 * Validated
 */
__device__ inline ulong ModMulFastConst(ulong a, ulong b, ulong modulus, ulong bInv) {
    //NativeInt q      = MultDHi(this->m_value, bInv.m_value);
    ulong q = __umul64hi(a, bInv);
    //NativeInt yprime = this->m_value * b.m_value - q * modulus.m_value;
    ulong yprime = a * b - q * modulus;
    //return SignedNativeInt(yprime) - SignedNativeInt(modulus.m_value) >= 0 ? yprime - modulus.m_value : yprime;
    return (long)yprime - (long)modulus >=0 ? yprime - modulus : yprime;
}

__device__ inline uint128_t Mul128(ulong a, ulong b) {
    return (uint128_t)a * (uint128_t)b;
}

/**
 * add two 64-bit number with carry out, c = a + b
 * @param a: operand 1
 * @param b: operand 2
 * @param c: c = a + b
 * @return 1 if overflow occurs, 0 otherwise
 */
__device__ inline ulong AdditionWithCarryOut(ulong a, ulong b, ulong& c) {
    c = a + b;
    return (c < a);
}

/**
 * check if adding two 64-bit number can cause overflow
 * @param a: operand 1
 * @param b: operand 2
 * @return 1 if overflow occurs, 0 otherwise
 */
__device__ inline ulong IsAdditionOverflow(ulong a, ulong b) {
    return (a + b) < b ? 1 : 0;
}

/**
 * Barrett reduction of 128-bit integer modulo 64-bit integer. Source: Menezes,
 * Alfred; Oorschot, Paul; Vanstone, Scott. Handbook of Applied Cryptography,
 * Section 14.3.3.
 * @param a: operand (128-bit)
 * @param m: modulus (64-bit)
 * @param mu: 2^128/modulus (128-bit)
 * @return result: 64-bit result = a mod m
 */
__device__ inline ulong BarrettUint128ModUint64(uint128_t a, ulong modulus, uint128_t mu) {
    ulong a_lo = (uint64_t)a;
    ulong a_hi = a >> 64;
    ulong mu_lo = (uint64_t)mu;
    ulong mu_hi = mu >> 64;

    // Compute intermediate values
    ulong left_hi = (Mul128(a_lo, mu_lo)) >> 64;
    uint128_t middle = Mul128(a_lo, mu_hi);
    ulong middle_lo = (uint64_t)middle;
    ulong middle_hi = middle >> 64;

    ulong carry, tmp1, tmp2;
    //carry = AdditionWithCarryOut(middle_lo, left_hi, tmp1);
    tmp1 = middle_lo + left_hi;
    carry = (tmp1 < middle_lo);
    tmp2 = middle_hi + carry;

    middle = Mul128(a_hi, mu_lo);
    middle_lo = (uint64_t)middle;
    middle_hi = middle >> 64;
    carry = IsAdditionOverflow(middle_lo, tmp1);
    left_hi = middle_hi + carry;

    tmp1 = a_hi * mu_hi + tmp2 + left_hi;
    ulong result = a_lo - tmp1 * modulus;

    result -= (result >= modulus) ? modulus : 0;

    return result;
}

#endif // SHARED_DEVICE_FUNCTIONS_CUH
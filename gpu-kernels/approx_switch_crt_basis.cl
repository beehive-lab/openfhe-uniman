typedef struct m_vectors_struct {
        unsigned long* data; // Pointer to dynamically allocated array
        unsigned long modulus;
}m_vectors_struct;

/*
   ulong2 -> [ulong, ulong]
               low    high
                0      1
                x      y
*/
unsigned long long mul128(unsigned long a, unsigned long b) {
    unsigned long long result = (unsigned long long)a * (unsigned long long)b;
    return result;
}
    /*ulong2 va = 0;
    ulong2 vb = 0;
    va[0] = a;
    vb[0] = b;
    return a * b;*/


/*ulong2 int128modInt64(ulong2 a, ulong b) {
    
    ulong a_hi = a[1];
    ulong a_lo = a[0];

    ulong result_hi = a_hi % b;
    ulong result_lo = a_lo % b;

    return (ulong2)(result_lo, result_hi);
}*/

ulong additionWithCarryOut(ulong a, ulong b, ulong c) {
    a += b;
    c = a;
    if (a < b)
        return 1;
    else
        return 0;
}

ulong isAdditionOverflow(ulong a, ulong b) {
    a += b;
    if (a < b)
        return 1;
    else
        return 0;
}

/*ulong barrettUint128ModUint64(ulong2 a, ulong modulus, ulong2 mu) {
    ulong result = 0, a_lo = 0, a_hi = 0, mu_lo = 0, mu_hi = 0, left_hi = 0, middle_lo = 0, middle_hi = 0, tmp1 = 0, tmp2 = 0, carry = 0;
    ulong2 middle;// = (0,0);

    a_lo = a[0];
    a_hi = a[1];
    mu_lo = mu[0];
    mu_hi = mu[1];

    left_hi = mul128(a_lo, mu_lo)[0];

    middle = mul128(a_lo, mu_hi);
    middle_lo = middle[0];
    middle_hi = middle[1];

    carry = additionWithCarryOut(middle_lo, left_hi, tmp1);

    tmp2 = middle_hi + carry;

    middle = mul128(a_hi, mu_lo);
    middle_lo = middle[0];
    middle_hi = middle[1];

    carry = isAdditionOverflow(middle_lo, tmp1);

    left_hi = middle_hi + carry;

    tmp1 = a_hi * mu_hi + tmp2 + left_hi;

    result = a_lo - tmp1 * modulus;

    while (result >= modulus)
        result -= modulus;

    return result;

}*/

//__kernel void approx_switch_crt_basis(__global m_vectors_struct* m_vectors, __global ulong* qhatinvmodq, __global ulong* qhatmodp, __local ulong2* local_sum, int SIZE_Q, int SIZE_P, __global ulong2* sum) {
__kernel void approx_switch_crt_basis(__global m_vectors_struct* m_vectors, __global ulong* qhatinvmodq, __global ulong* qhatmodp, int SIZE_Q, int SIZE_P, __global unsigned long* sum) {
    
    //const size_t ri = get_global_id(0);
    //uint local_addr = get_local_id(0);
    for (size_t ri = 0; ri < 10; ri++) {


        for (size_t i = 0; i < SIZE_Q; i++) {
            const ulong xi = m_vectors[i].data[ri];
            const ulong qi = m_vectors[i].modulus;

            // modular multiplication - formula: (a * b) mod m
            ulong xQHatInvModqi = (xi * qhatinvmodq[i]) % m_vectors[i].modulus;
            printf("[kernel]: [ri = %d, i= %d], xQHatInvModqi = %ld\n",xQHatInvModqi);

            //barrier(CLK_LOCAL_MEM_FENCE);
            for (size_t j = 0; j < SIZE_P; j++) {
                //ulong2
                //sum[(ri*SIZE_P) + j] += mul128(xQHatInvModqi, qhatmodp[i * SIZE_P + j]);
                sum[(ri*SIZE_P) + j] += 1;
                //local_sum[local_addr] += mul128(xQHatInvModqi, qhatmodp[i * SIZE_P + j]);
            }
        }
    }

    /*barrier(CLK_LOCAL_MEM_FENCE);
    if(get_local_id(0) == 0) {
        for(int i=0; i < get_local_size(0); i++) {
            for(int j = 0; j < SIZE_P; j++) {
                sum[j] += local_sum[i];
            }
        }
        //group_result[get_group_id(0)] = sum;
    }*/

    /*for (size_t j = 0; j < SIZE_P; j++) {
        const ulong pj = ans_m_vectors[j].modulus;
        //ans_m_vectors[j][ri] = barrettUint128ModUint64(sum[j], pj, )
    }*/

}
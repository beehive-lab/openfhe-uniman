//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2014-2022, NJIT, Duality Technologies Inc. and other contributors
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================

/*
  Simple example for BGVrns (integer arithmetic)
 */

#include "openfhe.h"
#include "gpu-acceleration/opencl_utils.h"
#include <chrono>
using namespace std::chrono;

using namespace lbcrypto;

void fillVectorWithRandomIntegers(std::vector<int64_t>& vec, int64_t N, int64_t min, int64_t max) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int64_t> distribution(min, max);

    for (int64_t i = 0; i < N; ++i) {
        vec.push_back(distribution(generator));
    }
}

int main() {
    ///// Opencl host code

    #define PROGRAM_FILE "/home/orion/Encrypt/openfhe-development/gpu-kernels/approx_switch_crt_basis.cl"
    //#define PROGRAM_FILE "/home/orion/Encrypt/openfhe-development/gpu-kernels/approx_switch_crt_basis.cl"
    #define KERNEL_FUNC "approx_switch_crt_basis"
    //#define KERNEL_FUNC "approx_switch_crt_basis"

    printf("init opencl\n");
    initializeKernel(PROGRAM_FILE, KERNEL_FUNC);
    printf("success!\n");

    /* Initialize data */
    for(int a = 0; a < ARRAY_SIZE; a++) {
        data[a] = 1.0f*a;
    }

    /* Build program */
    //printf("build program\n");
    //program = build_program(context, *devices, PROGRAM_FILE);
    //printf("success\n");

    //printf("config kernel\n");
    //configKernel(KERNEL_FUNC, 256, 128); //256, 128
    //printf("success!\n");
    /////

    // Sample Program: Step 1 - Set CryptoContext
    CCParams<CryptoContextBGVRNS> parameters;
    parameters.SetMultiplicativeDepth(12);
    parameters.SetPlaintextModulus(65537);

    CryptoContext<DCRTPoly> cryptoContext = GenCryptoContext(parameters);
    // Enable features that you wish to use
    cryptoContext->Enable(PKE);
    cryptoContext->Enable(KEYSWITCH);
    cryptoContext->Enable(LEVELEDSHE);

    // Sample Program: Step 2 - Key Generation

    // Initialize Public Key Containers
    KeyPair<DCRTPoly> keyPair;

    // Generate a public/private key pair
    keyPair = cryptoContext->KeyGen();

    // Generate the relinearization key
    printf("multiplication keygen start\n");
    cryptoContext->EvalMultKeyGen(keyPair.secretKey);
    printf("multiplication keygen end\n");

    // Generate the rotation evaluation keys
    cryptoContext->EvalRotateKeyGen(keyPair.secretKey, {1, 2, -1, -2});

    // Sample Program: Step 3 - Encryption

    // First plaintext vector is encoded
    std::vector<int64_t> vectorOfInts1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Plaintext plaintext1               = cryptoContext->MakePackedPlaintext(vectorOfInts1);
    // Second plaintext vector is encoded
    std::vector<int64_t> vectorOfInts2 = {3, 2, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Plaintext plaintext2               = cryptoContext->MakePackedPlaintext(vectorOfInts2);
    // Third plaintext vector is encoded
    std::vector<int64_t> vectorOfInts3 = {1, 2, 5, 2, 5, 6, 7, 8, 9, 10, 11, 12};
    Plaintext plaintext3               = cryptoContext->MakePackedPlaintext(vectorOfInts3);

    // The number of vectors you want to generate
    int64_t M = 5;

    // The vector size
    int64_t N = 3;//32768;

    // The number of multiplications
    int64_t Z = 1;

    // The range of random integers (e.g., between -100 and 100)
    int64_t min = 0;
    int64_t max = 100;

    // Generate random indices in the range [0, M]
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int64_t> indexDistribution(0, M - 1);

    // Create a vector of vectors to store the random integers
    std::vector<std::vector<int64_t>> vectorOfVectorsOfInts;
    std::vector<Plaintext> vectorOfPlaintexts;

    // Generate M vectors, each containing N random integers
    for (int64_t i = 0; i < M; ++i) {
        std::vector<int64_t> vectorOfInts;
        fillVectorWithRandomIntegers(vectorOfInts, N, min, max);
        vectorOfVectorsOfInts.push_back(vectorOfInts);
        vectorOfPlaintexts.push_back(cryptoContext->MakePackedPlaintext(vectorOfInts));
    }

    // The encoded vectors are encrypted
    auto ciphertext1 = cryptoContext->Encrypt(keyPair.publicKey, plaintext1);
    auto ciphertext2 = cryptoContext->Encrypt(keyPair.publicKey, plaintext2);
    auto ciphertext3 = cryptoContext->Encrypt(keyPair.publicKey, plaintext3);

    std::vector< Ciphertext<DCRTPoly> > vectorOfCiphertexts;
    for (int64_t i = 0; i < M; ++i) {
        vectorOfCiphertexts.push_back(cryptoContext->Encrypt(keyPair.publicKey, vectorOfPlaintexts[i]));
    }

    // Sample Program: Step 4 - Evaluation

    // Homomorphic additions
    auto ciphertextAdd12     = cryptoContext->EvalAdd(ciphertext1, ciphertext2);
    auto ciphertextAddResult = cryptoContext->EvalAdd(ciphertextAdd12, ciphertext3);

    // Homomorphic multiplications
    // modulus switching is done automatically because by default the modulus
    // switching method is set to AUTO (rather than MANUAL)
    // one dummy multiplication to initialize result variable
    auto ciphertextMultResult1 = cryptoContext->EvalMult(vectorOfCiphertexts[0], vectorOfCiphertexts[0]);
    printf("perform %ld multiplications\n", Z);
    auto start = high_resolution_clock::now();
    for (int64_t z = 0; z < Z; ++z) {
        int64_t index1 = indexDistribution(generator);
        int64_t index2 = indexDistribution(generator);
        ciphertextMultResult1 = cryptoContext->EvalMult(vectorOfCiphertexts[index1], vectorOfCiphertexts[index2]);
    }
    /*
    auto ciphertextMultResult2 = cryptoContext->EvalMult(ciphertextMultResult1, ciphertext1);
    auto ciphertextMultResult3 = cryptoContext->EvalMult(ciphertextMultResult2, ciphertext1);
    auto ciphertextMultResult4 = cryptoContext->EvalMult(ciphertextMultResult3, ciphertext1);
    auto ciphertextMultResult5 = cryptoContext->EvalMult(ciphertextMultResult4, ciphertext1);
    auto ciphertextMultResult6 = cryptoContext->EvalMult(ciphertextMultResult5, ciphertext1);
    auto ciphertextMultResult7 = cryptoContext->EvalMult(ciphertextMultResult6, ciphertext1);
    auto ciphertextMultResult8 = cryptoContext->EvalMult(ciphertextMultResult7, ciphertext1);
    auto ciphertextMultResult9 = cryptoContext->EvalMult(ciphertextMultResult8, ciphertext1);
    auto ciphertextMultResult10 = cryptoContext->EvalMult(ciphertextMultResult9, ciphertext1);
    auto ciphertextMultResult11 = cryptoContext->EvalMult(ciphertextMultResult10, ciphertext1);
    auto ciphertextMultResult12 = cryptoContext->EvalMult(ciphertextMultResult11, ciphertext1);
     */
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    auto total_mult = duration.count();
    //auto one_mult = total_mult / 10;

    //ciphertextMultResult10 = cryptoContext->EvalMultAndRelinearize(ciphertextMultResult, ciphertext1);
    //auto ciphertextMultResult11 = cryptoContext->EvalMultAndRelinearize(ciphertextMultResult9, ciphertext1);
    //ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    //ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertextMul12);
    // Homomorphic rotations
    auto ciphertextRot1 = cryptoContext->EvalRotate(ciphertext1, 1);
    auto ciphertextRot2 = cryptoContext->EvalRotate(ciphertext1, 2);
    auto ciphertextRot3 = cryptoContext->EvalRotate(ciphertext1, -1);
    auto ciphertextRot4 = cryptoContext->EvalRotate(ciphertext1, -2);

    // Sample Program: Step 5 - Decryption

    // Decrypt the result of additions
    Plaintext plaintextAddResult;
    cryptoContext->Decrypt(keyPair.secretKey, ciphertextAddResult, &plaintextAddResult);

    // Decrypt the result of multiplications
    Plaintext plaintextMultResult;
    cryptoContext->Decrypt(keyPair.secretKey, ciphertextMultResult1, &plaintextMultResult);

    // Decrypt the result of rotations
    Plaintext plaintextRot1;
    cryptoContext->Decrypt(keyPair.secretKey, ciphertextRot1, &plaintextRot1);
    Plaintext plaintextRot2;
    cryptoContext->Decrypt(keyPair.secretKey, ciphertextRot2, &plaintextRot2);
    Plaintext plaintextRot3;
    cryptoContext->Decrypt(keyPair.secretKey, ciphertextRot3, &plaintextRot3);
    Plaintext plaintextRot4;
    cryptoContext->Decrypt(keyPair.secretKey, ciphertextRot4, &plaintextRot4);

    plaintextRot1->SetLength(vectorOfInts1.size());
    plaintextRot2->SetLength(vectorOfInts1.size());
    plaintextRot3->SetLength(vectorOfInts1.size());
    plaintextRot4->SetLength(vectorOfInts1.size());

    std::cout << "Plaintext #1: " << plaintext1 << std::endl;
    std::cout << "Plaintext #2: " << plaintext2 << std::endl;
    std::cout << "Plaintext #3: " << plaintext3 << std::endl;

    // Output results
    std::cout << "\nResults of homomorphic computations" << std::endl;
    std::cout << "#1 + #2 + #3: " << plaintextAddResult << std::endl;
    std::cout << "#1 * #2 * #3: " << plaintextMultResult << std::endl;
    std::cout << "Left rotation of #1 by 1: " << plaintextRot1 << std::endl;
    std::cout << "Left rotation of #1 by 2: " << plaintextRot2 << std::endl;
    std::cout << "Right rotation of #1 by 1: " << plaintextRot3 << std::endl;
    std::cout << "Right rotation of #1 by 2: " << plaintextRot4 << std::endl;

    printf("time elapsed %ld msec.\n", total_mult);

    deallocateResources();

    return 0;
}

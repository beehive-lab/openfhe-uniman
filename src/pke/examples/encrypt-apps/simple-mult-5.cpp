#include <utils/timers.h>
#define PROFILE 1

#include "openfhe.h"

using namespace lbcrypto;

/**
 * Run configurations and input sizes:
 *
 * multiplications      = 5
 * plaintext modulus    = 65537
 * ring Dimension       = 16384     (is set internally by OpenFHE)
 * gpu blocks           = 16        (should be set manually at approx-switch-crt-basis.cu)
 * gpu threads          = 1024      (should be set manually at approx-switch-crt-basis.cu)
 *
 */

int main() {

    TimeVar appTimer, homOptimer;
    setNumOfOperations(5);
    TIC(appTimer);

    // Sample Program: Step 1 - Set CryptoContext
    CCParams<CryptoContextBGVRNS> parameters;
    parameters.SetMultiplicativeDepth(5);
    parameters.SetPlaintextModulus(65537);

    #if defined(WITH_CUDA)
    // Access the singleton instance of cudaDataUtils
    cudaDataUtils& cudaUtils = cudaDataUtils::getInstance();
    // Set GPU configuration - Note: suitable for RTX 3050
    cudaUtils.initialize(32, 512, 6, 16384, 2, 6, 7);
    #endif

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
    cryptoContext->EvalMultKeyGen(keyPair.secretKey);

    // Generate the rotation evaluation keys
    cryptoContext->EvalRotateKeyGen(keyPair.secretKey, {1, 2, -1, -2});

    // Sample Program: Step 3 - Encryption

    // First plaintext vector is encoded
    std::vector<int64_t> vectorOfInts1 = {1,1,1,1};
    Plaintext plaintext1               = cryptoContext->MakePackedPlaintext(vectorOfInts1);
    // Second plaintext vector is encoded
    std::vector<int64_t> vectorOfInts2 = {1,1,1,1};
    Plaintext plaintext2               = cryptoContext->MakePackedPlaintext(vectorOfInts2);

    // The encoded vectors are encrypted
    auto ciphertext1 = cryptoContext->Encrypt(keyPair.publicKey, plaintext1);
    auto ciphertext2 = cryptoContext->Encrypt(keyPair.publicKey, plaintext2);

    // Sample Program: Step 4 - Evaluation
    // 5 Homomorphic multiplication
    TIC(homOptimer);
    auto ciphertextMultResult = cryptoContext->EvalMult(ciphertext1, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    accumulateTimer(operationsTimer, TOC_NS(homOptimer));

    // Sample Program: Step 5 - Decryption

    // Decrypt the result of multiplications
    Plaintext plaintextMultResult;
    cryptoContext->Decrypt(keyPair.secretKey, ciphertextMultResult, &plaintextMultResult);

    std::cout << "Plaintext #1: " << plaintext1 << std::endl;
    std::cout << "Plaintext #2: " << plaintext2 << std::endl;

    // Output results
    std::cout << "\nResults of 5 homomorphic multiplications (1 * 1) : " << plaintextMultResult << std::endl;

    accumulateTimer(applicationTimer, TOC_NS(appTimer));

    printTimers();

    #if defined(WITH_CUDA)
    cudaUtils.destroy();
    #endif

    return 0;

}

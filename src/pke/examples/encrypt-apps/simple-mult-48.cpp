#include <utils/timers.h>

#include "openfhe.h"

using namespace lbcrypto;

/**
 * Run configurations and input sizes:
 *
 * multiplications      = 48
 * plaintext modulus    = 131072
 * cyclotomic order     = 131072    (?)
 * ring Dimension       = 65536     (is set internally by OpenFHE)
 * gpu blocks           = 64
 * gpu threads / block  = 512
 *
 */

int main() {

    TimeVar appTimer, homOptimer;
    setNumOfOperations(48);
    TIC(appTimer);

    // Sample Program: Step 1 - Set CryptoContext
    CCParams<CryptoContextBGVRNS> parameters;
    parameters.SetMultiplicativeDepth(48);
    parameters.SetPlaintextModulus(786433);

    #if defined(WITH_CUDA)
    // Access the singleton instance of cudaDataUtils
    cudaDataUtils& cudaUtils = cudaDataUtils::getInstance();
    // Set GPU configuration - Note: suitable for RTX 3050
    cudaUtils.initialize(128, 512, 50, 65536, 12, 49, 50);
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
    // 1 Homomorphic multiplication
    TIC(homOptimer);
    auto ciphertextMultResult = cryptoContext->EvalMult(ciphertext1, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);//10
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);//20
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);//30
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);//40
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    ciphertextMultResult = cryptoContext->EvalMult(ciphertextMultResult, ciphertext2);
    accumulateTimer(operationsTimer, TOC_MS(homOptimer));

    // Sample Program: Step 5 - Decryption

    // Decrypt the result of multiplications
    Plaintext plaintextMultResult;
    cryptoContext->Decrypt(keyPair.secretKey, ciphertextMultResult, &plaintextMultResult);

    std::cout << "Plaintext #1: " << plaintext1 << std::endl;
    std::cout << "Plaintext #2: " << plaintext2 << std::endl;

    // Output results
    std::cout << "\nResults of 48 homomorphic multiplications (1 * 1) : " << plaintextMultResult << std::endl;

    accumulateTimer(applicationTimer, TOC_MS(appTimer));

    printTimers();

    #if defined(WITH_CUDA)
    cudaUtils.destroy();
    #endif

    return 0;

}

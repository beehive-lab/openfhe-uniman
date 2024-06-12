#include "openfhe.h"

// header files needed for serialization
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/bgvrns/bgvrns-ser.h"

using namespace lbcrypto;

const std::string DATAFOLDER = "demoData";

//binary decision trees
typedef struct bdt
{
    std::vector<int64_t> root;
    bdt* left;
    bdt* right;
} bdt;

//adding a unique value to each slot of a vector
//std::vector<int> incr_indexes(std::vector<int> indexes)
//{
//std::vector<int> result = {};
//for(int i=0; i<(int)indexes.size(); i++)
//{
//result.push_back(indexes[i]+1);
//}
//return result;
//}

//constructing the list of the indexes for the nodes of a serialized tree
//std::vector<int> node_indexes(int depth)
//{
//if (depth == 1)
//{
//return {0};
//}
//else
//{
//std::vector<int> result = {0};
//std::vector<int> left = incr_indexes(node_indexes(depth-1));
//std::vector<int> right = incr_indexes(incr_indexes(node_indexes(depth-1)));
//for(int i=0; i<(int)left.size(); i++)
//{
//result.push_back(incr_indexes(node_indexes(depth-1))[i]);
//}
//for(int i=0; i<(int)left.size(); i++)
//{
//result.push_back(incr_indexes(incr_indexes(node_indexes(depth-1)))[i]);
//}
//return result;
//}
//}

//splitting a vector in two (in order to build a tree)
std::vector<std::vector<std::vector<int64_t>>> split(std::vector<std::vector<int64_t>> tags, int depth)
{
    std::vector<std::vector<std::vector<int64_t>>> result;
    result.push_back({});
    result.push_back({});

    int cpt = 1;

	
    for(int i=1; i<depth; i++)
    {
        for(int j=0; j<pow(2, i-1); j++)
        {
            (result[0]).push_back(tags[cpt]);
            cpt++;
        }
        for(int j=0; j<pow(2, i-1); j++)
        {
            (result[1]).push_back(tags[cpt]);
            cpt++;
        }
    }
	
    return result;
}

//building a tree from a vector
bdt build_tree(std::vector<std::vector<int64_t>> tags, int depth)
{
    bdt tree;
    tree.root = tags[0];
    std::cout << tree.root << std::endl;
    tree.left = new bdt();
    tree.right = new bdt();
    if(depth>1)
    {
        *(tree.left) = build_tree(split(tags, depth)[0], depth-1);
        *(tree.right) = build_tree(split(tags, depth)[1], depth-1);
    }
    else
    {
        tree.left = NULL;
        tree.right = NULL;
    }
    return tree;
}

//binary decision trees encoded as plaintexts
typedef struct bdt_pt
{
    Plaintext root;
    bdt_pt* left;
    bdt_pt* right;
} bdt_pt;

//encrypted binary decision trees
typedef struct bdt_ct
{
    Ciphertext<DCRTPoly> root;
    bdt_ct* left;
    bdt_ct* right;
} bdt_ct;

//encoding a binary decision tree
bdt_pt bdt_encode(CryptoContext<DCRTPoly> cc, bdt tree)
{
    bdt_pt result;
    result.root = cc->MakePackedPlaintext(tree.root);
    result.left = new bdt_pt();
    result.right = new bdt_pt();
    if(tree.left!=NULL)
    {
        *(result.left) = bdt_encode(cc, *(tree.left));
    }
    else
    {
        result.left = NULL;
    }
    if(tree.right!=NULL)
    {
        *(result.right) = bdt_encode(cc, *(tree.right));
    }
    else
    {
        result.right=NULL ;
    }
    return result;
}

//encryption of a binary decision tree
bdt_ct bdt_encrypt(CryptoContext<DCRTPoly> cc, bdt_pt tree, const PublicKey<DCRTPoly> pk)
{
    bdt_ct result;
    result.root = cc->Encrypt(pk, tree.root);
    result.left = new bdt_ct();
    result.right = new bdt_ct();
    if(tree.left!=NULL)
    {
        *(result.left) = bdt_encrypt(cc, *(tree.left), pk);
    }
    else
    {
        result.left = NULL;
    }
    if(tree.right!=NULL)
    {
        *(result.right) = bdt_encrypt(cc, *(tree.right), pk);
    }
    else
    {
        result.right = NULL;
    }
    return result;
}

//subfunction for serializing of an encrypted bdt in a recursive way (TODO mettre a jour)
//void ebdt_serialize_switched(bdt_ct tree, int depth, int k)
//{
//std::vector<int> indexes = node_indexes(depth);
	
//if (!Serial::SerializeToFile(DATAFOLDER + "/" + "ciphertext" + std::to_string(indexes[k]) + ".txt", tree.root, SerType::BINARY)) {
//std::cerr << "Error writing serialization of node " << indexes[k] << "to ciphertext" << indexes[k] << ".txt" << std::endl;
//}
//std::cout << "serialized ciphertext " << indexes[k] << std::endl;
    
//int cpt = k;
//if(tree.left!=NULL)
//{
//cpt++;
//ebdt_serialize_switched(*(tree.left), depth, cpt);
//}
//if(tree.right!=NULL)
//{
//cpt++;
//ebdt_serialize_switched(*(tree.left), depth, cpt);
//}
		
//}

// subfunction for recursive serialization of an encrypted bdt (from depth-first search)
int ebdt_serialize_switched(bdt_ct tree, std::string name, int i)
{
    if (!Serial::SerializeToFile(DATAFOLDER + "/" + name + std::to_string(i) + ".txt", tree.root, SerType::BINARY)) {
        std::cerr << "Error writing serialization of node " << i << "to ciphertext" << i << ".txt" << std::endl;
    }
    std::cout << "serialized ciphertext " << i << std::endl;
    i++;
    
    if(tree.left !=NULL)
    {
        i = ebdt_serialize_switched(*(tree.left), name, i);
    }
    //i++; (?)
    
    if(tree.right !=NULL)
    {
        i = ebdt_serialize_switched(*(tree.right), name, i);
    }
    //i++; (?)
    
    return i;
}

// serialization of an encrypted bdt from depth-first search (ebdt_serialize_switched with i=0 and no output value)
void ebdt_serialize(bdt_ct tree, std::string name)
{
    ebdt_serialize_switched(tree, name, 0);
}

// homomorphic node-per-node substraction of two encrypted BDTs (the result being a new encrypted BDT)
bdt_ct bdt_evalSub(CryptoContext<DCRTPoly> cc, bdt_ct tree1, bdt_ct tree2)
{
    bdt_ct result;
    result.root = cc->EvalSub(tree1.root, tree2.root);
    result.left = new bdt_ct();
    result.right = new bdt_ct();
    if((tree1.left!=NULL) && (tree2.left!=NULL))
    {
        *(result.left) = bdt_evalSub(cc, *(tree1.left), *(tree2.left));
    }
    else
    {
        result.left = NULL;
    }
    if((tree1.right!=NULL) && (tree2.right!=NULL))
    {
        *(result.right) = bdt_evalSub(cc, *(tree1.right), *(tree2.right));
    }
    else
    {
        result.right = NULL;
    }
    return result;
}

// integers from 0 to N-1 in a binary form (this represents all the paths of a BT (without tags) of depth N)
std::vector<std::vector<int64_t>> abstract_paths(int N)
{
    std::vector<std::vector<int64_t>> result;
    for(int i=0;i<pow(2,N);i++)
    {
        result.push_back({});
        for(int j=0;j<N;j++)
        {
            result[i].push_back((int)(i/pow(2, N-1-j))%2);
        }
    }
    return result;
}

// the same in reverse order
std::vector<std::vector<int64_t>> abstract_reversePaths(int N)
{
    std::vector<std::vector<int64_t>> result;
    for(int i=0;i<pow(2,N);i++)
    {
        result.push_back({});
        for(int j=0;j<N;j++)
        {
            result[i].push_back((int)(1-(int)(i/pow(2, N-1-j))%2));
        }
    }
    return result;
}

// encoding abstract_paths(N)
std::vector<std::vector<Plaintext>> encoded_abstract_paths(CryptoContext<DCRTPoly> cc, int N)
{
    std::vector<std::vector<Plaintext>> result;
    for(int i=0;i<pow(2,N);i++)
    {
        result.push_back({});
        for(int j=0;j<N;j++)
        {
            result[i].push_back(cc->MakePackedPlaintext({abstract_paths(N)[i][j]}));
        }
    }
    return result;
}

// the same in reverse order
std::vector<std::vector<Plaintext>> encoded_abstract_reversePaths(CryptoContext<DCRTPoly> cc, int N)
{
    std::vector<std::vector<Plaintext>> result;
    for(int i=0;i<pow(2,N);i++)
    {
        result.push_back({});
        for(int j=0;j<N;j++)
        {
            result[i].push_back(cc->MakePackedPlaintext({abstract_reversePaths(N)[i][j]}));
        }
    }
    return result;
}

// encrypting abstract_paths(N)
std::vector<std::vector<Ciphertext<DCRTPoly>>> encrypted_abstract_paths(CryptoContext<DCRTPoly> cc, int N, const PublicKey<DCRTPoly> pk)
{
    std::vector<std::vector<Ciphertext<DCRTPoly>>> result;
    for(int i=0;i<pow(2,N);i++)
    {
        result.push_back({});
        for(int j=0;j<N;j++)
        {
            result[i].push_back(cc->Encrypt(pk, encoded_abstract_paths(cc, N)[i][j]));
        }
    }
    return result;
}

// the same in reverse order
std::vector<std::vector<Ciphertext<DCRTPoly>>> encrypted_abstract_reversePaths(CryptoContext<DCRTPoly> cc, int N, const PublicKey<DCRTPoly> pk)
{
    std::vector<std::vector<Ciphertext<DCRTPoly>>> result;
    for(int i=0;i<pow(2,N);i++)
    {
        result.push_back({});
        for(int j=0;j<N;j++)
        {
            result[i].push_back(cc->Encrypt(pk, encoded_abstract_reversePaths(cc, N)[i][j]));
        }
    }
    return result;
}

// decrypting a vector of paths
//std::vector<std::vector<Plaintext>> v_decrypt(CryptoContext<DCRTPoly> cc, std::vector<std::vector<Ciphertext<DCRTPoly>>> ciphertexts, const PrivateKey<DCRTPoly> sk)
//{
//std::vector<std::vector<Plaintext>> result;
//for(long unsigned int i=0; i<ciphertexts.size();i++)
//{
//result.push_back({});
//for(long unsigned int j=0;j<ciphertexts[i].size();j++)
//{
//result[i].push_back(Plaintext());
//cc->Decrypt(sk, ciphertexts[i][j], &(result[i][j]));
//}
//}
//return result;
//}

//// packing a vector of ciphertexts
//Ciphertext<DCRTPoly> pack_ct(CryptoContext<DCRTPoly> cc, std::vector<Ciphertext<DCRTPoly>> ciphertexts, int N)
//{
//	Ciphertext<DCRTPoly> result = ciphertexts[N-1];
//	for(int i=0; i<N-1; i++)
//	{
//		result = cc->EvalAdd(cc->EvalRotate(result, -1), ciphertexts[N-2-i]);
//	}
//	return result;
//}



// paths of an encrypted bdt (as a vector of encrypted paths)
std::vector<std::vector<Ciphertext<DCRTPoly>>> bdt_evalPaths(CryptoContext<DCRTPoly> cc, bdt_ct tree, int depth)
{
    std::vector<std::vector<Ciphertext<DCRTPoly>>> result(pow(2, depth));
    std::vector<Ciphertext<DCRTPoly>> path(depth);
    for(int i=0; i<pow(2,depth);i++)
    {
        bdt_ct subtree = tree;
        for(int j=0; j<depth;j++)
        {
            result[i].push_back(Ciphertext<DCRTPoly>());
            result[i][j]=subtree.root;
            if(((subtree.right)!=NULL) && ((subtree.right)!=NULL))
            {
                subtree = ((abstract_paths(depth)[i][j]) ? (*(subtree.right)) : (*(subtree.left)));
            }
        }
    }
    return result;
}

//std::vector<std::vector<Ciphertext<DCRTPoly>>> bdt_evalReductedPaths(CryptoContext<DCRTPoly> cc, bdt_ct tree, int depth)
//{
//std::vector<std::vector<Ciphertext<DCRTPoly>>> result(pow(2, depth-1));
//std::vector<Ciphertext<DCRTPoly>> path(depth);
//std::vector<std::vector<Ciphertext<DCRTPoly>>> paths = bdt_evalPaths(cc, tree, depth);
//for(int i=0; i<pow(2,depth-1);i++)
//{
//for(int j=0; j<depth;j++)
//{
//result[i].push_back(paths[2*i][j]);
//}
//}
//return result;
//}

// encrypted paths of a bdt with value 1 at each node
//std::vector<std::vector<Ciphertext<DCRTPoly>>> encrypted_paths_with_1_everywhere(CryptoContext<DCRTPoly> cc, int N, const PublicKey<DCRTPoly> pk)
//{
//Ciphertext<DCRTPoly> encrypted_un = cc->Encrypt(pk, cc->MakePackedPlaintext({1}));
//std::vector<std::vector<Ciphertext<DCRTPoly>>> result(pow(2, N));
//for(int i=0; i<pow(2,N);i++)
//{
//result[i] = std::vector<Ciphertext<DCRTPoly>>(N);
//for(int j=0; j<N; j++)
//{
//result[i][j] = encrypted_un;
//}
//}
//return result;
//}

//encrypted result (with some slots with value -1 instead of 1) before the final homomorphic multiplications
std::vector<std::vector<Ciphertext<DCRTPoly>>> encrypted_result_before_mult(CryptoContext<DCRTPoly> cc, bdt_ct tree, bdt_ct data, int depth, const PublicKey<DCRTPoly> pk)
{
    int number_of_leaves = pow(2, depth);
    bdt_ct bdt_deltas = bdt_evalSub(cc, data, tree);
    std::vector<std::vector<Ciphertext<DCRTPoly>>> deltas = bdt_evalPaths(cc, bdt_deltas, depth);
    std::vector<std::vector<Ciphertext<DCRTPoly>>> result(number_of_leaves);
    //std::vector<std::vector<Ciphertext<DCRTPoly>>> otherTerm(number_of_leaves);
    //for(int i=0; i<number_of_leaves; i++)
    //{
    //otherTerm[i] = std::vector<Ciphertext<DCRTPoly>>(depth);
    //for(int j=0; j<depth; j++)
    //{
    //   otherTerm[i][j] = cc->EvalSub(encrypted_abstract_paths(cc, depth, pk)[i][j], encrypted_paths_with_1_everywhere(cc, depth, pk)[i][j]);
    //}
    //}
    //	std::vector<std::vector<Ciphertext<DCRTPoly>>> result(number_of_leaves);
    for(int i=0; i<number_of_leaves; i++)
    {
        result[i] = std::vector<Ciphertext<DCRTPoly>>(depth);
        for(int j=0; j<depth; j++)
        {
            result[i][j] = cc->EvalSub(cc->EvalMult(deltas[i][j], deltas[i][j]), encrypted_abstract_reversePaths(cc, depth, pk)[i][j]);
        }
    }
    return result;
} 



//////////////////////////////////////
// IL RESTE A TESTER A PARTIR DE LA //
//////////////////////////////////////
	
// homomorphic evaluation of the product of the elements of a vector
Ciphertext<DCRTPoly> evalGlobalProd(CryptoContext<DCRTPoly> cc, std::vector<Ciphertext<DCRTPoly>> ciphertexts)
{
    Ciphertext<DCRTPoly> result = ciphertexts[0];
    for(unsigned int i=1; i<ciphertexts.size(); i++)
    {
        result = cc->EvalMult(result, ciphertexts[i]);
    }
    return result;
}

// evalGlobalProd on each element of a vector
std::vector<Ciphertext<DCRTPoly>> v_evalGlobalProd(CryptoContext<DCRTPoly> cc, std::vector<std::vector<Ciphertext<DCRTPoly>>> ciphertexts)
{
    std::vector<Ciphertext<DCRTPoly>> result;
    for(unsigned int i=0; i<ciphertexts.size(); i++)
    {
        result.push_back(evalGlobalProd(cc, ciphertexts[i]));
    }
    return result;
}

//final products. The output is a vector filled with 1 or -1 at the index associated to the output leave and 0 everywhere else.
std::vector<Ciphertext<DCRTPoly>> encrypted_result_after_mult(CryptoContext<DCRTPoly> cc, bdt_ct tree, bdt_ct data, int depth, const PublicKey<DCRTPoly> pk)
{
    return v_evalGlobalProd(cc, encrypted_result_before_mult(cc, tree, data, depth, pk));
}

//encoded powers of 2 in a reverse order (to get numerical values from their binary form)
std::vector<Plaintext> powersOf2(CryptoContext<DCRTPoly> cc, int N, const PublicKey<DCRTPoly> pk)
{
    std::vector<Plaintext> result;
    for(int i=0; i<N; i++)
    {
        result.push_back(cc->MakePackedPlaintext({(int) pow(2, N-1-i)}));
    }
    return result;
}

//encypted final result
Ciphertext<DCRTPoly> encrypted_result(CryptoContext<DCRTPoly> cc, bdt_ct tree, bdt_ct data, int depth, const PublicKey<DCRTPoly> pk)
{
    // faudra retirer sk des variables dans la version final
    std::vector<std::vector<Ciphertext<DCRTPoly>>> bin_result_at_some_slot;
    std::vector<std::vector<Ciphertext<DCRTPoly>>> eap = encrypted_abstract_paths(cc, depth, pk);
    std::vector<Ciphertext<DCRTPoly>> eram = encrypted_result_after_mult(cc, tree, data, depth, pk);
    for(int i=0; i<pow(2, depth); i++)
    {
        bin_result_at_some_slot.push_back({});
        for(int j=0; j<depth; j++)
        {
            bin_result_at_some_slot[i].push_back(cc->EvalMult(eap[i][j], eram[i]));
        }
    }
	
    //std::cout << "bin result at some slot" << std::endl;
    //Plaintext P;
    //for(int i=0; i<8; i++)
    //{
    //for(int j=0;j<3;j++)
    //{
    //cc->Decrypt(sk, bin_result_at_some_slot[i][j], &P);
    //std::cout <<  P << ";";
    //}
    //std::cout << std::endl;
    //}
	
    std::vector<Ciphertext<DCRTPoly>> bin_result;
    for(int j=0; j<depth; j++)
    {
        bin_result.push_back(cc->Encrypt(pk, cc->MakePackedPlaintext({0})));
    }
    for(int i=0; i<pow(2,depth); i++)
    {
        for(int j=0; j<depth; j++)
        {
            bin_result[j] = cc->EvalAdd(bin_result[j], bin_result_at_some_slot[i][j]);
        }
    }
	
    //std::cout << "bin result" << std::endl;
    //for(int j=0; j<3; j++)
    //{
    //cc->Decrypt(sk, bin_result[j], &P);
    //std::cout <<  P << ";";
    //}
    //std::cout << std::endl;
	 
    //std::cout << "flag2" << std::endl;
	
    //(here the -1 values are turned into 1 values in order to have each value at 0 or 1)
    for(int j=0; j<depth; j++)
    {
        bin_result[j] = cc->EvalMult(bin_result[j], bin_result[j]);
    }
	
    //std::cout << "flag3" << std::endl;
	
    //std::cout << "new bin result" << std::endl;
    //for(int j=0; j<3; j++)
    //{
    //cc->Decrypt(sk, bin_result[j], &P);
    //std::cout <<  P << ";";
    //}
    //std::cout << std::endl;
	
    Ciphertext<DCRTPoly> result = cc->Encrypt(pk, cc->MakePackedPlaintext({0}));
    std::vector<Ciphertext<DCRTPoly>> acc;
    acc = std::vector<Ciphertext<DCRTPoly>>();
    for(int j=0; j<depth; j++)
    {
        acc.push_back(cc->EvalMult(bin_result[j], powersOf2(cc, depth, pk)[j]));
        result = cc->EvalAdd(acc[j], result);
    }
	
    //std::cout << "flag4" << std::endl;
    //cc->Decrypt(sk, result, &P);
    //std::cout << "result : " << P << std::endl;
	
    return result;
}
		


/////////////////////////////////////////////
//                                         //
//               |MAIN|                    //
//                                         //
/////////////////////////////////////////////

int main()
{
    //cryptocontext setting
    CCParams<CryptoContextBGVRNS> parameters;
    parameters.SetMultiplicativeDepth(8);
    parameters.SetPlaintextModulus(65537);
      
    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
      
    // Serialize cryptocontext
    if (!Serial::SerializeToFile(DATAFOLDER + "/cryptocontext.txt", cc, SerType::BINARY)) {
        std::cerr << "Error writing serialization of the crypto context to "
                     "cryptocontext.txt"
                  << std::endl;
        return 1;
    }
    std::cout << "The cryptocontext has been serialized." << std::endl;
      
    //key generation
    KeyPair<DCRTPoly> keyPair;
    keyPair = cc->KeyGen();
    const PublicKey<DCRTPoly> pk = keyPair.publicKey;
    const PrivateKey<DCRTPoly> sk = keyPair.secretKey;
      
    // Serialize the secret key
    if (!Serial::SerializeToFile(DATAFOLDER + "/key-private.txt", keyPair.secretKey, SerType::BINARY)) {
        std::cerr << "Error writing serialization of private key to key-private.txt" << std::endl;
        return 1;
    }
    std::cout << "The secret key has been serialized." << std::endl;
    
    //cc->EvalRotateKeyGen(sk, {-1});
      
    cc->EvalMultKeyGen(sk);
      
    /////////////////////////////////////////////////////////////////////////////
    // ENCRYPTING AND DECRYPTING THE ABSTRACT PATHS OF AN UNTAGGED BINARY TREE //
    /////////////////////////////////////////////////////////////////////////////
      
    //std::cout << "abstract paths of the (untagged) binary tree of depth 3" << std::endl;
      
    ////printing all the abstract paths of the (untagged) BT of depth 3
    //std::vector<std::vector<int64_t>> paths3 = abstract_paths(3);
      
    //for(int i=0;i<8;i++)
    //{
    //for(int j=0;j<3;j++)
    //{
    //std::cout << paths3[i][j] << ";";
    //}
    //std::cout << std::endl;
    //}
      
    ////in reverse order
    //std::cout << "reverse table : " << std::endl;
    //std::vector<std::vector<int64_t>> reversePaths3 = abstract_reversePaths(3);
      
    //for(int i=0;i<8;i++)
    //{
    //for(int j=0;j<3;j++)
    //{
    //std::cout << reversePaths3[i][j] << ";";
    //}
    //std::cout << std::endl;
    //}
    ////printing all the encoded abstract paths of the (untagged) BT of depth 3
    //std::vector<std::vector<Plaintext>> encoded_paths3 = encoded_abstract_paths(cc, 3);
    //for(int i=0;i<8;i++)
    //{
    //for(int j=0;j<3;j++)
    //{
    //std::cout << encoded_paths3[i][j] << ";" << ";";
    //}
    //std::cout << std::endl;
    //}
	  
    ////retrieving all the abstract paths of the (untagged) BT of depth 3 from a ciphertext
    //std::vector<std::vector<Ciphertext<DCRTPoly>>> encrypted_paths3 = encrypted_abstract_paths(cc, 3, pk);
    //std::vector<std::vector<Plaintext>> result_dec_paths3 = v_decrypt(cc, encrypted_paths3, sk);
    //for(int i=0;i<8;i++)
    //{
    //for(int j=0;j<3;j++)
    //{
    //std::cout << result_dec_paths3[i][j] << ";" << std::endl;
    //}
    //}
	  
    //tests for splitting vectors and turning them into trees
    std::vector<std::vector<int64_t>> tags = {{1}, {2}, {3}, {4}, {5}, {6}, {7}};
    std::vector<std::vector<int64_t>> left_tags = split(tags, 3)[0];
    std::vector<std::vector<int64_t>> right_tags = split(tags, 3)[1];
    bdt tags_in_tree = build_tree(tags, 3);
	  
    // printing the left tags and the right tags
    std::cout << left_tags[0] << left_tags[1] << left_tags[2] << std::endl;
    std::cout << right_tags[0] << right_tags[1] << right_tags[2] << std::endl;
	  
    // printing the nodes
    //std::cout << tags_in_tree.root << (tags_in_tree.left)->root << (tags_in_tree.right)->root << (tags_in_tree.left)->left->root << (tags_in_tree.left)->right->root << (tags_in_tree.right)->left->root << (tags_in_tree.right)->right->root << std::endl;
	  
    //////////////////////////////////////////////////////
    // ENCRYPTING AND DECRYPTING A BINARY DECISION TREE //
    //////////////////////////////////////////////////////
	  
    //constructing a tree
    bdt tree;
      
    //(nodes)
    tree.root = {1};
      
    tree.left = new bdt();
    tree.right = new bdt();
    (tree.left)->root = {1};
    (tree.right)->root = {0};
      
      
    (tree.left)->left = new bdt();
    (tree.left)->right = new bdt();
    (tree.right)->left = new bdt();
    (tree.right)->right = new bdt();
    (tree.left)->left->root = {0};
    (tree.left)->right->root = {1};
    (tree.right)->left->root = {0};
    (tree.right)->right->root = {1};
       
    //(leaves)
    (tree.left)->left->left = NULL;
    (tree.left)->left->right = NULL;
    (tree.left)->right->left = NULL;
    (tree.left)->right->right = NULL;
    (tree.right)->left->left = NULL;
    (tree.right)->left->right = NULL;
    (tree.right)->right->left = NULL;
    (tree.right)->right->right = NULL;
      
    std::cout << "nodes of the binary decision tree (step by step and from left to right)" << std::endl;
      
    // printing the nodes
    std::cout << tree.root << (tree.left)->root << (tree.right)->root << (tree.left)->left->root << (tree.left)->right->root << (tree.right)->left->root << (tree.right)->right->root << std::endl;
      
    //encoding
    bdt_pt encoded_tree = bdt_encode(cc, tree);
      
    ////printing the encoded tree
    //std::cout << encoded_tree.root << (encoded_tree.left)->root << (encoded_tree.right)->root << (encoded_tree.left)->left->root << (encoded_tree.left)->right->root << (encoded_tree.right)->left->root << (encoded_tree.right)->right->root << std::endl;
      
    //encrypting
    bdt_ct encrypted_tree = bdt_encrypt(cc, encoded_tree, pk);
      
    //testing node_indexes
    //for(int i=0; i<7; i++)
    //{
    //	  std::cout << node_indexes(3)[i] << "; " << std::endl;
    //}
      
    //serialization
    ebdt_serialize(encrypted_tree, "encrypted_tree");
      
    ////decrypting
    //bdt_pt result = bdt_decrypt(cc, encrypted_tree, sk);
      
    ////printing the result
    //std::cout << result.root << (result.left)->root << (result.right)->root << (result.left)->left->root << (result.left)->right->root << (result.right)->left->root << (result.right)->right->root << std::endl;
      
      
      
    ///////////////////////////////////////
    // EVALUATING A BINARY DECISION TREE //
    ///////////////////////////////////////
      
    //(to be removed)//
    //std::vector<Ciphertext<DCRTPoly>> testpaths = {cc->Encrypt(pk, cc->MakePackedPlaintext({3})), cc->Encrypt(pk, cc->MakePackedPlaintext({5})), cc->Encrypt(pk, cc->MakePackedPlaintext({1}))};
    //Ciphertext<DCRTPoly> testctpacking = pack_ct(cc, testpaths, 3);
    //Plaintext testdecpaths;
    //cc->Decrypt(sk, testctpacking, &testdecpaths);

      
    //// constructing the encrypted paths of the tree
    //std::vector<std::vector<Ciphertext<DCRTPoly>>> encrypted_paths = bdt_evalReductedPaths(cc, encrypted_tree, 3);
      
    //// decrypting them
    //std::vector<std::vector<Plaintext>> paths= v_decrypt(cc, encrypted_paths , sk);
    //std::cout << "paths of the tree" << std::endl;
    //for(int i=0;i<4;i++)
    //{
    //for(int j=0;j<3;j++)
    //{
    //std::cout << paths[i][j] << ";";
    //}
    //std::cout << std::endl;
    //}
	  
	  
    //std::vector<std::vector<Plaintext>> ev_paths= v_decrypt(cc, encrypted_paths, sk);
    //std::cout << "paths of the decision tree" << std::endl;
    //for(int i=0;i<8;i++)
    //{
    //for(int j=0;j<3;j++)
    //{
    //std::cout << ev_paths[i][j] << ";";
    //}
    //std::cout << std::endl;
    //}
	  
    // constructing input data as another tree
    bdt data;
      
    //nodes
    data.root = {0};
      
    data.left = new bdt();
    data.right = new bdt();
    (data.left)->root = {1};
    (data.right)->root = {1};
      
      
    (data.left)->left = new bdt();
    (data.left)->right = new bdt();
    (data.right)->left = new bdt();
    (data.right)->right = new bdt();
    (data.left)->left->root = {0};
    (data.left)->right->root = {1};
    (data.right)->left->root = {1};
    (data.right)->right->root = {0};
       
    //(leaves)
    (data.left)->left->left = NULL;
    (data.left)->left->right = NULL;
    (data.left)->right->left = NULL;
    (data.left)->right->right = NULL;
    (data.right)->left->left = NULL;
    (data.right)->left->right = NULL;
    (data.right)->right->left = NULL;
    (data.right)->right->right = NULL;
	
    std::cout << "nodes of the binary data tree (step by step and from left to right)" << std::endl;
      
    // printing the nodes
    std::cout << data.root << (data.left)->root << (data.right)->root << (data.left)->left->root << (data.left)->right->root << (data.right)->left->root << (data.right)->right->root << std::endl;
	  
    //encoding
    bdt_pt encoded_data = bdt_encode(cc, data);
	  
    //encrypting
    bdt_ct encrypted_data = bdt_encrypt(cc, encoded_data, pk);
	  
    //// constructing the encrypted paths of the tree
    //std::vector<std::vector<Ciphertext<DCRTPoly>>> encrypted_data_paths = bdt_evalReductedPaths(cc, encrypted_data, 3);
    //std::cout << "ok here" << std::endl;
      
    //// decrypting them
    //std::vector<std::vector<Plaintext>> data_paths= v_decrypt(cc, encrypted_data_paths , sk);
    //std::cout << "paths of the data tree" << std::endl;
    //for(int i=0;i<4;i++)
    //{
    //for(int j=0;j<3;j++)
    //{
    //std::cout << data_paths[i][j] << ";";
    //}
    //std::cout << std::endl;
    //}
	  
    ////homomorphic evaluation of the first tree with the data tree in input
    //bdt_ct encrypted_comparisons = bdt_evalSub(cc, encrypted_data, encrypted_tree);
	  
    //decription of paths with value 1 everywhere
    //std::vector<std::vector<Ciphertext<DCRTPoly>>> encrypted_un = encrypted_paths_with_1_everywhere(cc, 3, pk);
    //std::vector<std::vector<Plaintext>> un = v_decrypt(cc, encrypted_un, sk);
    //std::cout << "paths with value 1 everywhere" << std::endl;
    //for(int i=0;i<8;i++)
    //{
    //	  for(int j=0;j<3;j++)
    //	  {
    //	     std::cout << un[i][j] << ";";
    //	  }
    //	  std::cout << std::endl;
    //}
	  
    //result before the final multiplications
    //std::vector<std::vector<Ciphertext<DCRTPoly>>> encrypted_resBeforeMult = encrypted_result_before_mult(cc, encrypted_tree, encrypted_data, 3, pk);
    //std::vector<std::vector<Plaintext>> resBeforeMult = v_decrypt(cc, encrypted_resBeforeMult, sk);
    //std::cout << "result before the final multiplications" << std::endl;
    //for(int i=0;i<8;i++)
    //{
    //for(int j=0;j<3;j++)
    //{
    //std::cout << resBeforeMult[i][j] << ";";
    //}
    //std::cout << std::endl;
    //}
	  
    //std::cout << "ok here" << std::endl;
    //std::vector<Plaintext> test1;
    //std::vector<Ciphertext<DCRTPoly>> products;
    //for(int i=0; i<8; i++)
    //{
    //test1.push_back(Plaintext());
    //products.push_back(evalGlobalProd(cc, encrypted_resBeforeMult[i]));
    //std::cout << "ok " << i << "; ";
    //cc->Decrypt(sk, products[i], &(test1[i]));
    //std::cout << "ok" << i << ";";
    //}
    //std::cout << "test 1 output : ";
    //for(int i=0; i<8; i++)
    //{
    //std::cout << test1[i] << " ; ";
    //}
    //std::cout << std::endl;
	  
    //std::vector<Plaintext> test2;
    //std::vector<Ciphertext<DCRTPoly>> allproducts = v_evalGlobalProd(cc, encrypted_resBeforeMult);
    //for(int i=0; i<8; i++)
    //{
    //test2.push_back(Plaintext());
    //cc->Decrypt(sk, allproducts[i], &(test2[i]));
    //}
    //std::cout << "test 2 output : ";
    //for(int i=0; i<8; i++)
    //{
    //std::cout << test2[i] << " ; ";
    //}
    //std::cout << std::endl;
	  
    //std::vector<Ciphertext<DCRTPoly>> encrypted_resAfterMult = encrypted_result_after_mult(cc, encrypted_tree, encrypted_data, 3, pk);
    //std::vector<Plaintext> test3;
    //for(int i=0; i<8; i++)
    //{
    //test3.push_back(Plaintext());
    //cc->Decrypt(sk, encrypted_resAfterMult[i], &(test3[i]));
    //}
    //std::cout << "test 3 output : ";
    //for(int i=0; i<8; i++)
    //{
    //std::cout << test3[i] << " ; ";
    //}
    //std::cout << std::endl;
	  
	  
    ////powers of 2 in reverse order
    //std::vector<Plaintext> reversePowersOf2 = powersOf2(cc, 3, pk);
    ////std::vector<Plaintext> reversePowersOf2 = {};
    ////for (int j=0; j<3; j++)
    ////{
    ////	  reversePowersOf2.push_back(Plaintext());
    ////	  cc->Decrypt(sk, encrypted_reversePowersOf2[j], &(reversePowersOf2[j]));
    ////}
	  
    //std::cout << "powers of 2 in reverse order" << std::endl;
    //for(int j=0;j<3;j++)
    //{
    //std::cout << reversePowersOf2[j] << ";";
    //}
    //std::cout << std::endl;
	  
    //final result
    //faudra returer l'argument sk dans la version finale
    Ciphertext<DCRTPoly> output_ciphertext = encrypted_result(cc, encrypted_tree, encrypted_data, 3, pk);
    Plaintext final_output;
    cc->Decrypt(sk, output_ciphertext, &final_output);
    std::cout << "OUTPUT VALUE : " << final_output << std::endl;
	  
    //////////////////////////////
    //////////////////////////////
      
    //main return value
    return 0;
}
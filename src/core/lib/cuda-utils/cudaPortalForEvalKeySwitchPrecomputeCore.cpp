#include "math/hal.h"
#include "lattice/poly.h"
#include <cstdint> // for uint32_t type

#include "cuda-utils/cudaPortalForEvalKeySwitchPrecomputeCore.h"

#include <cuda-utils/cudaPortalForApproxModDown.h>

namespace lbcrypto {

using PolyType = PolyImpl<NativeVector>;

// constructor impl
cudaPortalForEvalKeySwitchPrecomputeCore::cudaPortalForEvalKeySwitchPrecomputeCore() {

}

}

#ifndef UNMARSHAL_DATA_BATCH_H
#define UNMARSHAL_DATA_BATCH_H

struct UnmarshalWorkDataBatchParams {
    std::vector<lbcrypto::PolyImpl<NativeVector>>*  ans_m_vectors;
    ulong*                                          host_ans_m_vectors;
    uint32_t                                        size;
    uint32_t                                        i;
    uint32_t                                        ptrOffset;

    UnmarshalWorkDataBatchParams(std::vector<lbcrypto::PolyImpl<NativeVector>>* a,
                                 ulong* host_ans_m_vectors,
                                 uint32_t size,
                                 uint32_t idx,
                                 uint32_t offset)
        : ans_m_vectors(a), host_ans_m_vectors(host_ans_m_vectors), size(size), i(idx), ptrOffset(offset) {}
};

#endif //UNMARSHAL_DATA_BATCH_H

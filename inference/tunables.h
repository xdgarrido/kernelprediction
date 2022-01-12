/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <string>
#include <vector>
#include <assert.h>
#include "classifier.h"

#define IGEMM_GTC_TUNABLE_FMA_TYPE_MAC              "mac"
#define IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS            "dlops"
#define IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS           "xdlops"
#define IGEMM_GTC_TUNABLE_FMA_TYPE_NA               "fma_na"

bool tunable_is_valid_fwd(vector<int> vec, string tensor_layout, string precision);
bool tunable_is_valid_bwd(vector<int> vec, string tensor_layout, string precision);
bool tunable_is_valid_wrw(vector<int> vec, string tensor_layout, string precision);
void print_batch_file(const std::string& file_name, vector<vector<int>> predicted_codes, string conv_type, string precision, string layout);
int verify_prediction(vector<vector<int>> predicted_codes, vector<int> codes, int separation_idx, int no_of_candidates);

typedef struct {
    std::string tensor_layout;
    int gemm_m_per_block;
    int gemm_n_per_block;
    int gemm_k_per_block;
    std::string fma_type;
    union {
        struct {
            int gemm_m_per_thread;
            int gemm_m_level0_cluster;
            int gemm_m_level1_cluster;
            int gemm_n_per_thread;
            int gemm_n_level0_cluster;
            int gemm_n_level1_cluster;
            int dummy;
        };
        struct {
            int wave_tile_m;
            int wave_step_m;
            int wave_repeat_m;
            int wave_tile_n;
            int wave_step_n;
            int wave_repeat_n;
            int wave_tile_k;
        };
    };
    int tensor_a_pass_through;
    int tensor_b_pass_through;
    std::vector<int> tensor_a_thread_lengths;
    std::vector<int> tensor_a_cluster_lengths;
    std::vector<int> tensor_b_thread_lengths;
    std::vector<int> tensor_b_cluster_lengths;
    std::string direction;
    std::string precision;
    int nxb;
    int nxe;
    int gemm_m_unmerge_cluster;
    int gemm_n_unmerge_cluster;
    int gemm_k_unmerge_cluster;
    int multihead;
    int source_access_order;
    int vector_store;
    int gemm_k_global_split;
    int merge_e;
} igemm_gtc_tunable_t_x;

template <typename T>
T utility_gcd_x(T x, T y)
{
    if (x == y || x == 0)
    {
        return y;
    }
    else if (y == 0)
    {
        return x;
    }
    else if (x > y)
    {
        return utility_gcd_x(x - y, y);
    }
    else
    {
        return utility_gcd_x(x, y - x);
    }
}

template <typename T>
T utility_integer_divide_floor_x(T x, T y)
{
    return x / y;
}

template <typename T>
T utility_integer_divide_ceil_x(T x, T y)
{
    return (x + y - 1) / y;
}

template <typename T>
T utility_max_x(T x, T y)
{
    return x > y ? x : y;
}

template <typename T>
T utility_min_x(T x, T y)
{
    return x < y ? x : y;
}

static inline std::string
utility_int_list_to_string_x(const std::vector<int> list) {
    std::string enc;
    for (int i = 0; i < list.size(); i++) {
        enc.append(std::to_string(list[i]));
        if (i != (list.size() - 1))
            enc.append("x");
    }
    return enc;
}

static inline int utility_next_pow2_x(int n) {
    if (n == 0)
        return 1;
    if ((n & (n - 1)) == 0)
        return n;
    while ((n & (n - 1)) > 0)
        n &= (n - 1);
    return n << 1;
}

static inline int utility_string_to_data_byte_x(std::string precision)
{
    if (precision == "fp32")
        return 4;
    if (precision == "fp16" || precision == "bf16")
        return 2;
    assert(false);
    return 1;
}

static inline std::string
encode_kernel_name(const igemm_gtc_tunable_t_x *tunable) {
    int gcn_arch = 908; // hardwired to MI100's
    auto tensor_layout = tunable->tensor_layout;
    auto gemm_m_per_block = tunable->gemm_m_per_block;
    auto gemm_n_per_block = tunable->gemm_n_per_block;
    auto gemm_k_per_block = tunable->gemm_k_per_block;
    auto fma_type = tunable->fma_type;
    // auto gemm_m_per_thread        = tunable->gemm_m_per_thread;
    // auto gemm_m_level0_cluster    = tunable->gemm_m_level0_cluster;
    // auto gemm_m_level1_cluster    = tunable->gemm_m_level1_cluster;
    // auto gemm_n_per_thread        = tunable->gemm_n_per_thread;
    // auto gemm_n_level0_cluster    = tunable->gemm_n_level0_cluster;
    // auto gemm_n_level1_cluster    = tunable->gemm_n_level1_cluster;
    auto tensor_a_pass_through = tunable->tensor_a_pass_through;
    auto tensor_b_pass_through = tunable->tensor_b_pass_through;
    auto tensor_a_thread_lengths = tunable->tensor_a_thread_lengths;
    auto tensor_a_cluster_lengths = tunable->tensor_a_cluster_lengths;
    auto tensor_b_thread_lengths = tunable->tensor_b_thread_lengths;
    auto tensor_b_cluster_lengths = tunable->tensor_b_cluster_lengths;
    auto direction = tunable->direction;
    auto precision = tunable->precision;
    auto nxb = tunable->nxb;
    auto nxe = tunable->nxe;
    auto gemm_m_unmerge_cluster = tunable->gemm_m_unmerge_cluster;
    auto gemm_n_unmerge_cluster = tunable->gemm_n_unmerge_cluster;
    auto gemm_k_unmerge_cluster = tunable->gemm_k_unmerge_cluster;
    auto source_access_order = tunable->source_access_order;
    auto multihead = tunable->multihead;
    auto vector_store = tunable->vector_store;
    auto gemm_k_global_split = tunable->gemm_k_global_split;
    auto merge_e = tunable->merge_e;

    //static int gcn_arch = -1;
    //if (gcn_arch == -1) {
    //    hipDeviceProp_t dev_prop;
    //    hipDevice_t dev;
    //    HIP_CALL(hipGetDevice(&dev));
    //    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
    //    gcn_arch = dev_prop.gcnArch;
    //}

    std::string kernel_name = std::string("igemm_") + direction + "_";
    if (tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_MAC)
        kernel_name += "gtcm_";
    else if (tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS)
        kernel_name += "gtc_";
    else if (tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS) {
        if (gcn_arch == 908)
            kernel_name += "gtcx_";
        else if (gcn_arch == 910)
            kernel_name += "gtcx2_";
    }

    kernel_name += tensor_layout + std::string("_") + precision +
        std::string("_bx") + std::to_string(nxb) +
        std::string("_ex") + std::to_string(nxe) +
#if USE_SOURCE_ACCESS_ENCODING_KERNEL_NAME
        std::string("_sa") + std::to_string(source_access_order) + "_";
#else
        "_";
#endif

    kernel_name += std::string("bt") +
        std::to_string(gemm_m_per_block) + "x" +
        std::to_string(gemm_n_per_block) + "x" +
        std::to_string(gemm_k_per_block) + "_";

    if (tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_MAC || tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS) {
        auto gemm_m_per_thread = tunable->gemm_m_per_thread;
        auto gemm_m_level0_cluster = tunable->gemm_m_level0_cluster;
        auto gemm_m_level1_cluster = tunable->gemm_m_level1_cluster;
        auto gemm_n_per_thread = tunable->gemm_n_per_thread;
        auto gemm_n_level0_cluster = tunable->gemm_n_level0_cluster;
        auto gemm_n_level1_cluster = tunable->gemm_n_level1_cluster;
        assert(gemm_m_per_block % (gemm_m_per_thread * gemm_m_level0_cluster * gemm_m_level1_cluster) == 0);
        assert(gemm_n_per_block % (gemm_n_per_thread * gemm_n_level0_cluster * gemm_n_level1_cluster) == 0);
        int gemm_m_repeat = gemm_m_per_block / (gemm_m_per_thread * gemm_m_level0_cluster * gemm_m_level1_cluster);
        int gemm_n_repeat = gemm_n_per_block / (gemm_n_per_thread * gemm_n_level0_cluster * gemm_n_level1_cluster);

        int thread_tile_m = gemm_m_repeat * gemm_m_per_thread;
        int thread_tile_n = gemm_n_repeat * gemm_n_per_thread;
        kernel_name += std::string("tt") +
            std::to_string(thread_tile_m) + "x" +
            std::to_string(thread_tile_n) + "_" +
            "gm" +
            std::to_string(gemm_m_repeat) + "x" +
            std::to_string(gemm_m_level0_cluster) + "x" +
            std::to_string(gemm_m_level1_cluster) + "_" +
            "gn" +
            std::to_string(gemm_n_repeat) + "x" +
            std::to_string(gemm_n_level0_cluster) + "x" +
            std::to_string(gemm_n_level1_cluster) + "_";
    }
    else if (tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS) {
        kernel_name += std::string("wt") + std::to_string(tunable->wave_tile_m) + "x" + std::to_string(tunable->wave_tile_n) + "x" + std::to_string(tunable->wave_tile_k) + "_" +
            "ws" + std::to_string(tunable->wave_step_m) + "x" + std::to_string(tunable->wave_step_n) + "_" +
            "wr" + std::to_string(tunable->wave_repeat_m) + "x" + std::to_string(tunable->wave_repeat_n) + "_";
    }

    kernel_name +=
        "ta" + utility_int_list_to_string_x(tensor_a_thread_lengths) + "_" +
        utility_int_list_to_string_x(tensor_a_cluster_lengths) + "_" +
        "tb" + utility_int_list_to_string_x(tensor_b_thread_lengths) + "_" +
        utility_int_list_to_string_x(tensor_b_cluster_lengths);
    // printf("[%s]\n",kernel_name.c_str());
    if (tensor_a_pass_through)
        kernel_name += std::string("_pta");
    if (tensor_b_pass_through)
        kernel_name += std::string("_ptb");
    if (gemm_m_unmerge_cluster)
        kernel_name += std::string("_mc");
    if (gemm_n_unmerge_cluster)
        kernel_name += std::string("_nc");
    if (gemm_k_unmerge_cluster)
        kernel_name += std::string("_kc");
    if (multihead)
        kernel_name += std::string("_mh");
    if (merge_e)
        kernel_name += std::string("_me");
    // when split in gemmk, we need call atomic add function
    if (vector_store)
        kernel_name += std::string("_vs") + std::to_string(vector_store);
    if (gemm_k_global_split > 0)
        kernel_name += std::string("_gkgs");
    return kernel_name;
}

#define KERNEL_PARAMETER_DIMENSION SEP_IDX

static inline void vec2tunable(vector<int> vec, string tensor_layout, string precision, string direction, igemm_gtc_tunable_t_x *tunable)
{
    
    tunable->gemm_m_per_block = vec[0+ KERNEL_PARAMETER_DIMENSION];
    tunable->gemm_n_per_block = vec[1 + KERNEL_PARAMETER_DIMENSION];
    tunable->gemm_k_per_block = vec[2 + KERNEL_PARAMETER_DIMENSION];
    tunable->wave_tile_m = vec[3 + KERNEL_PARAMETER_DIMENSION];
    tunable->wave_tile_n = vec[4 + KERNEL_PARAMETER_DIMENSION];
    tunable->wave_tile_k = vec[5 + KERNEL_PARAMETER_DIMENSION];
    tunable->wave_step_m = vec[6 + KERNEL_PARAMETER_DIMENSION];
    tunable->wave_step_n = vec[7 + KERNEL_PARAMETER_DIMENSION];
    tunable->wave_repeat_m = vec[8 + KERNEL_PARAMETER_DIMENSION];
    tunable->wave_repeat_n = vec[9 + KERNEL_PARAMETER_DIMENSION];

    for (int i = 0; i < 4; i++)
    {
        tunable->tensor_a_thread_lengths.push_back(vec[10 + i + KERNEL_PARAMETER_DIMENSION]);
    }
    for (int i = 0; i < 4; i++)
    {
        tunable->tensor_a_cluster_lengths.push_back(vec[14 + i + KERNEL_PARAMETER_DIMENSION]);
    }
    for (int i = 0; i < 4; i++)
    {
        tunable->tensor_b_thread_lengths.push_back(vec[18 + i + KERNEL_PARAMETER_DIMENSION]);
    }
    for (int i = 0; i < 4; i++)
    {
        tunable->tensor_b_cluster_lengths.push_back(vec[22 + i + KERNEL_PARAMETER_DIMENSION]);
    }


    tunable->nxb = vec[26 + KERNEL_PARAMETER_DIMENSION];
    tunable->nxe = vec[27 + KERNEL_PARAMETER_DIMENSION];
    tunable->gemm_k_global_split = vec[28 + KERNEL_PARAMETER_DIMENSION];
    tunable->merge_e = vec[29 + KERNEL_PARAMETER_DIMENSION];
    tunable->tensor_a_pass_through = vec[30 + KERNEL_PARAMETER_DIMENSION];

    // these are constants equal to 0
    tunable->gemm_m_per_thread=0;
    tunable->gemm_m_level0_cluster=0;
    tunable->gemm_m_level1_cluster=0;
    tunable->gemm_n_per_thread=0;
    tunable->gemm_n_level0_cluster=0;
    tunable->gemm_n_level1_cluster=0;
    tunable->dummy=0;

    tunable->gemm_m_unmerge_cluster = 0;
    tunable->gemm_n_unmerge_cluster = 0;
    tunable->gemm_k_unmerge_cluster = 0;

    tunable->vector_store = 0;

    
    // non constants depending of classifier
    tunable->tensor_layout = tensor_layout;
    tunable->fma_type = IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS;
    tunable->tensor_b_pass_through = 0;
    tunable->direction = direction;
    tunable->precision = precision; 

    int default_source_access_order = tunable->direction == "fwd" ? 1 : 0;
    tunable->source_access_order = default_source_access_order;

    int default_mh = tunable->direction == "bwd" && tunable->tensor_layout == "nhwc" && tunable->nxe != 0 ? 1 : 0;
    tunable->multihead = default_mh;
    
}
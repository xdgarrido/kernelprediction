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

#include "tunables.h"
static inline int utility_string_to_data_byte(std::string precision)
{
    if(precision == "fp32")
        return 4;
    if(precision == "fp16" || precision == "bf16")
        return 2;
    assert(false);
    return 1;
}
int split_batch_size(vector<int> vec)
    {
        int n = vec[0];
        int c = vec[1];
        int hi = vec[2];
        int wi = vec[3];
        int k = vec[4];
        int y = vec[5];
        int x = vec[6];
        int stride_h = vec[7];
        int stride_w = vec[8];
        int dilation_h = vec[9];
        int dilation_w = vec[10];
        int pad_h = vec[11];
        int pad_w = vec[12];
        int ho = vec[13];
        int wo = vec[14];
        int group = 1;
        
        int data_byte = utility_string_to_data_byte("fp32");
        size_t image_size_input = static_cast<size_t>(c) * hi * wi * data_byte;
        size_t image_size_output = static_cast<size_t>(k) * ho * wo * data_byte;
        size_t size_4g = 0xffffffffUL;
        if(image_size_input >= size_4g || image_size_output >= size_4g)
            return 0;

        size_t image_size = image_size_input >= image_size_output ? image_size_input : image_size_output;
        size_t splited_n = size_4g / image_size;

        // round up splits, we must match
        // 1. splited_n * image_size < size_4g
        // 2. n % splited_n == 0
        // if(splited_n >= n)
        //     return 1;
        assert(splited_n != 0);
        while(splited_n >= 1){
            // printf("n:%d, splited_n:%d\n", n, splited_n);
            if(n % splited_n == 0)
                break;
            splited_n--;
        }

        assert(splited_n * image_size < size_4g && n % splited_n == 0);
        return ((int)(n / splited_n));
    }

#if 0  // original tunable_is_valid from conv driver
bool tunable_is_valid(const args_t *arg,
                          const igemm_gtc_tunable_t *tunable,
                          const driverDataType_t& data_type)
    {
        int hi = arg->get_int("in_h");
        int wi = arg->get_int("in_w");
        int n = arg->get_int("batchsize");
        int k = arg->get_int("out_channels");
        int c = arg->get_int("in_channels");

        int stride_h = arg->get_int("conv_stride_h");
        int stride_w = arg->get_int("conv_stride_w");
        int dilation_h = arg->get_int("dilation_h");
        int dilation_w = arg->get_int("dilation_w");
        int pad_h = arg->get_int("pad_h");
        int pad_w = arg->get_int("pad_w");
        int y = arg->get_int("fil_h");
        int x = arg->get_int("fil_w");
        int ho = conv_out_size(hi, pad_h, dilation_h, y, stride_h);
        int wo = conv_out_size(wi, pad_w, dilation_w, x, stride_w);
        int group = arg->get_int("group_count");

        assert(c % group == 0 && k % group == 0);

        std::string precision = tunable->precision;
        //std::cout << std::endl;
        if(precision == "fp16"){
            //std::cout << "is same type=" << std::is_same<gpu_data_type, float16>::value << std::endl;
            if(data_type != driverHalf){
                return false;
            }
        }
        else if(precision == "fp32"){
            //std::cout << "is same type=" << std::is_same<gpu_data_type, float>::value << std::endl;
            if(data_type != driverFloat){
                return false;
            }
        }
        else
        {
            std::cout << std::endl;
            std::cout << precision << std::endl;
            return false;
        }
        int splits = split_batch_size(arg, tunable);
        if(splits == 0){
            printf("image size (c*h*w) is bigger than 4g, which is not supported now\n");
            return false;
        }
        n = n/splits;   // split batch size here

        int gemm_m_per_block         = tunable->gemm_m_per_block;
        int gemm_n_per_block         = tunable->gemm_n_per_block;
        int gemm_k_per_block         = tunable->gemm_k_per_block;

        int nxe                      = tunable->nxe;
        int nxb                      = tunable->nxb;
        int b                        = nxe == 0 ? (ho * wo) : ((ho * wo + nxb - 1) / nxb) * nxb;   // pad to nxb modulo when nxe != 0

        int gemm_m = ((k/group + gemm_m_per_block -1)/gemm_m_per_block) * gemm_m_per_block;
        int gemm_n                   = n * b;
        int gemm_k                   = (c / group) * y * x;

        bool unit_conv = (x==1)&&(y==1)&&(stride_h==1)&&(stride_w==1)&&(dilation_h==1)&&(dilation_w==1)&&(pad_h==0)&&(pad_w==0);

        // support pad to modulo, hence only check when nxe is 0
        if((gemm_n % gemm_n_per_block != 0) || (gemm_m % gemm_m_per_block != 0))
            return false;

        // if(gemm_k % gemm_k_per_block != 0)
            // return false;

        if(gemm_n_per_block % tunable->nxb != 0){
            //printf("tunable_is_valid false: gemm_n_per_block%tunable->nxb!=0, gemm_n_per_block is %d, tunable->nxb is %d\n", gemm_n_per_block, tunable->nxb);
            return false;
        }

        if(n % (gemm_n_per_block / tunable->nxb) != 0){
            //printf("tunable_is_valid false: n%(gemm_n_per_block/tunable->nxb)!=0, gemm_n_per_block is %d, tunable->nxb is %d\n", gemm_n_per_block, tunable->nxb);
            return false;
        }

        if((nxe == 0) && ((b % tunable->nxb != 0) || (gemm_k % gemm_k_per_block != 0))){
            return false;
        }

        if((nxe == 0) && !unit_conv){
            return false;
        }

        // input vector load limitation, n1b
        if(tunable->tensor_b_thread_lengths[3] > 1 && (
            !unit_conv ||
            unit_conv && (hi * wi) % tunable->tensor_b_thread_lengths[3] != 0)) {
            return false;
        }

        // weight vector load limitation, c1e
        if(tunable->tensor_a_thread_lengths[1] > 1 &&
                gemm_k % tunable->tensor_a_thread_lengths[1] != 0){
            return false;
        }

        // if tb_c1e > 1, only 1x1 case is runable, it can not check gemm_k_padding either.
        if(tunable->tensor_b_thread_lengths[1] > 1 && ((pad_h !=0 || pad_w != 0)||( x !=1 || y != 1)||(gemm_k % gemm_k_per_block != 0))){
            return false;
        }

        // if t_c0 > 1, need to check gemmk per block
        if(tunable->tensor_b_thread_lengths[0] > 1 && (gemm_k % gemm_k_per_block != 0)){
            return false;
        }

        // let's check the next configuration even though this configuration is applicable
        // if (mayHaveBiggerN1bClusterSize(gemm_m, gemm_n, tunable) )
            // return(false); 

        return true;
    }
#endif 

bool tunable_is_valid(vector<int> vec)
{

    int n = vec[0];
    int c = vec[1];
    int hi = vec[2];
    int wi = vec[3];
    int k = vec[4];
    int y = vec[5];
    int x = vec[6];
    int stride_h = vec[7];
    int stride_w = vec[8];
    int dilation_h = vec[9];
    int dilation_w = vec[10];
    int pad_h = vec[11];
    int pad_w = vec[12];
    int ho = vec[13];
    int wo = vec[14];
    int group = 1;

    int gemm_m_per_block = vec[15];
    int gemm_n_per_block = vec[16];
    int gemm_k_per_block = vec[17];
    int nxb = vec[41];
    int nxe = vec[42];
    //int elapsed_time = par[43];
    int tensor_a_thread_lengths[4];
    int tensor_b_thread_lengths[4];

    for (int i = 0; i < 4; i++)
    {
        tensor_a_thread_lengths[i] = vec[i + 25];
    }

    for (int i = 0; i < 4; i++)
    {
        tensor_b_thread_lengths[i] = vec[i + 33];
    }


    // Data acquisition order from conv_driver

    //printf("%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
    //    n, c, hi, wi, k, y, x, stride_h, stride_w, dilation_h, dilation_w,
    //    pad_h, pad_w, ho, wo);

    // printf(";%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
    //    gemm_m_per_block, gemm_n_per_block, gemm_k_per_block, wave_tile_m, wave_tile_n, wave_tile_k,
    //    wave_step_m, wave_step_n, wave_repeat_m, wave_repeat_n); (15 - 24)

    //for (int i = 0; i < tensor_a_thread_lengths.size(); i++)  (25,26,27,28)
    //{
    //    printf(",%d", tensor_a_thread_lengths[i]);
    //}
    //for (int i = 0; i < tensor_a_cluster_lengths.size(); i++) (29,30,31,32)
    //{
    //    printf(",%d", tensor_a_cluster_lengths[i]);
    //}
    //for (int i = 0; i < tensor_b_thread_lengths.size(); i++)  (33,34,35,36)
    //{
    //    printf(",%d", tensor_b_thread_lengths[i]);
    //}
    //for (int i = 0; i < tensor_b_cluster_lengths.size(); i++)  (37,38,39,40)
    //{
    //    printf(",%d", tensor_b_cluster_lengths[i]);
    //}

    //printf(",%d", nxb); (41)
    //printf(",%d", nxe); (42)
    //printf(",%d\n", (int)(1000 * elapsed_time));

    int splits = split_batch_size(vec);
    if(splits == 0){
            printf("image size (c*h*w) is bigger than 4g, which is not supported now\n");
            return false;
    }
    n = n/splits;   // split batch size here


    int b                        = nxe == 0 ? (ho * wo) : ((ho * wo + nxb - 1) / nxb) * nxb;   // pad to nxb modulo when nxe != 0

    int gemm_m = ((k/group + gemm_m_per_block -1)/gemm_m_per_block) * gemm_m_per_block;
    int gemm_n                   = n * b;
    int gemm_k                   = (c / group) * y * x;

    bool unit_conv = (x==1)&&(y==1)&&(stride_h==1)&&(stride_w==1)&&(dilation_h==1)&&(dilation_w==1)&&(pad_h==0)&&(pad_w==0);

        // support pad to modulo, hence only check when nxe is 0
    if((gemm_n % gemm_n_per_block != 0) || (gemm_m % gemm_m_per_block != 0))
        return false;

        // if(gemm_k % gemm_k_per_block != 0)
            // return false;

    if(gemm_n_per_block % nxb != 0){
            //printf("tunable_is_valid false: gemm_n_per_block%tunable->nxb!=0, gemm_n_per_block is %d, tunable->nxb is %d\n", gemm_n_per_block, tunable->nxb);
            return false;
    }

    if(n % (gemm_n_per_block / nxb) != 0){
            //printf("tunable_is_valid false: n%(gemm_n_per_block/tunable->nxb)!=0, gemm_n_per_block is %d, tunable->nxb is %d\n", gemm_n_per_block, tunable->nxb);
            return false;
    }

    if((nxe == 0) && ((b % nxb != 0) || (gemm_k % gemm_k_per_block != 0))){
            return false;
    }

    if((nxe == 0) && !unit_conv){
            return false;
    }

    // input vector load limitation, n1b
    if(tensor_b_thread_lengths[3] > 1 && (
            !unit_conv ||
            unit_conv && (hi * wi) % tensor_b_thread_lengths[3] != 0)) {
            return false;
    }

        // weight vector load limitation, c1e
    if(tensor_a_thread_lengths[1] > 1 &&
                gemm_k % tensor_a_thread_lengths[1] != 0){
            return false;
    }

        // if tb_c1e > 1, only 1x1 case is runable, it can not check gemm_k_padding either.
    if(tensor_b_thread_lengths[1] > 1 && ((pad_h !=0 || pad_w != 0)||( x !=1 || y != 1)||(gemm_k % gemm_k_per_block != 0))){
            return false;
    }

        // if t_c0 > 1, need to check gemmk per block
    if(tensor_b_thread_lengths[0] > 1 && (gemm_k % gemm_k_per_block != 0)){
            return false;
    }

        // let's check the next configuration even though this configuration is applicable
        // if (mayHaveBiggerN1bClusterSize(gemm_m, gemm_n, tunable) )
            // return(false); 

    return true;
}

bool tunable_is_valid2(vector<int> vec, vector<int> par)
{

    
    int n = vec[0];
    int c = vec[1];
    int hi = vec[2];
    int wi = vec[3];
    int k = vec[4];
    int y = vec[5];
    int x = vec[6];
    int stride_h = vec[7];
    int stride_w = vec[8];
    int dilation_h = vec[9];
    int dilation_w = vec[10];
    int pad_h = vec[11];
    int pad_w = vec[12];
    int ho = vec[13];
    int wo = vec[14];
    int group = 1;

    int gemm_m_per_block = par[15];
    int gemm_n_per_block = par[16];
    int gemm_k_per_block = par[17];
    int nxb = par[41];
    int nxe = par[42];
    //int elapsed_time = par[43];
    int tensor_a_thread_lengths[4];
    int tensor_b_thread_lengths[4];


    //printf("%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
    //    n, c, hi, wi, k, y, x, stride_h, stride_w, dilation_h, dilation_w,
    //    pad_h, pad_w, ho, wo);

    for (int i = 0; i < 4; i++)
    {
        tensor_a_thread_lengths[i] = par[i + 25];
    }

    for (int i = 0; i < 4; i++)
    {
        tensor_b_thread_lengths[i] = par[i + 33];
    }

    // Data acquisition order from conv_driver

    //printf("%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
    //    n, c, hi, wi, k, y, x, stride_h, stride_w, dilation_h, dilation_w,
    //    pad_h, pad_w, ho, wo);

    // printf(";%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
    //    gemm_m_per_block, gemm_n_per_block, gemm_k_per_block, wave_tile_m, wave_tile_n, wave_tile_k,
    //    wave_step_m, wave_step_n, wave_repeat_m, wave_repeat_n); (15 - 24)

    //for (int i = 0; i < tensor_a_thread_lengths.size(); i++)  (25,26,27,28)
    //{
    //    printf(",%d", tensor_a_thread_lengths[i]);
    //}
    //for (int i = 0; i < tensor_a_cluster_lengths.size(); i++) (29,30,31,32)
    //{
    //    printf(",%d", tensor_a_cluster_lengths[i]);
    //}
    //for (int i = 0; i < tensor_b_thread_lengths.size(); i++)  (33,34,35,36)
    //{
    //    printf(",%d", tensor_b_thread_lengths[i]);
    //}
    //for (int i = 0; i < tensor_b_cluster_lengths.size(); i++)  (37,38,39,40)
    //{
    //    printf(",%d", tensor_b_cluster_lengths[i]);
    //}

    //printf(",%d", nxb); (41)
    //printf(",%d", nxe); (42)
    //printf(",%d\n", (int)(1000 * elapsed_time));
    int splits = split_batch_size(vec);
    if(splits == 0){
            printf("image size (c*h*w) is bigger than 4g, which is not supported now\n");
            return false;
    }
    n = n/splits;   // split batch size here


    int b                        = nxe == 0 ? (ho * wo) : ((ho * wo + nxb - 1) / nxb) * nxb;   // pad to nxb modulo when nxe != 0

    int gemm_m = ((k/group + gemm_m_per_block -1)/gemm_m_per_block) * gemm_m_per_block;
    int gemm_n                   = n * b;
    int gemm_k                   = (c / group) * y * x;

    bool unit_conv = (x==1)&&(y==1)&&(stride_h==1)&&(stride_w==1)&&(dilation_h==1)&&(dilation_w==1)&&(pad_h==0)&&(pad_w==0);

        // support pad to modulo, hence only check when nxe is 0
    if((gemm_n % gemm_n_per_block != 0) || (gemm_m % gemm_m_per_block != 0))
        return false;

        // if(gemm_k % gemm_k_per_block != 0)
            // return false;

    if(gemm_n_per_block % nxb != 0){
            //printf("tunable_is_valid false: gemm_n_per_block%tunable->nxb!=0, gemm_n_per_block is %d, tunable->nxb is %d\n", gemm_n_per_block, tunable->nxb);
            return false;
    }

    if(n % (gemm_n_per_block / nxb) != 0){
            //printf("tunable_is_valid false: n%(gemm_n_per_block/tunable->nxb)!=0, gemm_n_per_block is %d, tunable->nxb is %d\n", gemm_n_per_block, tunable->nxb);
            return false;
    }

    if((nxe == 0) && ((b % nxb != 0) || (gemm_k % gemm_k_per_block != 0))){
            return false;
    }

    if((nxe == 0) && !unit_conv){
            return false;
    }

    // input vector load limitation, n1b
    if(tensor_b_thread_lengths[3] > 1 && (
            !unit_conv ||
            unit_conv && (hi * wi) % tensor_b_thread_lengths[3] != 0)) {
            return false;
    }

        // weight vector load limitation, c1e
    if(tensor_a_thread_lengths[1] > 1 &&
                gemm_k % tensor_a_thread_lengths[1] != 0){
            return false;
    }

        // if tb_c1e > 1, only 1x1 case is runable, it can not check gemm_k_padding either.
    if(tensor_b_thread_lengths[1] > 1 && ((pad_h !=0 || pad_w != 0)||( x !=1 || y != 1)||(gemm_k % gemm_k_per_block != 0))){
            return false;
    }

        // if t_c0 > 1, need to check gemmk per block
    if(tensor_b_thread_lengths[0] > 1 && (gemm_k % gemm_k_per_block != 0)){
            return false;
    }

        // let's check the next configuration even though this configuration is applicable
        // if (mayHaveBiggerN1bClusterSize(gemm_m, gemm_n, tunable) )
            // return(false); 

    return true;
}
void print_batch_file(const std::string& file_name, vector<vector<int>> predicted_codes)
{

    if (predicted_codes.size() != 0)
    {
    ofstream scope;

    scope.open(file_name, std::ios_base::app);


    scope << "./out/conv_driver.exe"  \
        << " conv " \
        << " -n " << predicted_codes[0][0]  /* batch */ \
        << " -c " << predicted_codes[0][1]  /* input chanels */ \
        << " -H " << predicted_codes[0][2]  /* height */\
        << " -W " << predicted_codes[0][3]  /* width */ \
        << " -k " << predicted_codes[0][4]  /* output channels */ \
        << " -y " << predicted_codes[0][5]  /* kernel height */ \
        << " -x " << predicted_codes[0][6]  /* kernel width */ \
        << " -u " << predicted_codes[0][7]  /* stride h */ \
        << " -v " << predicted_codes[0][8]  /* stride w */ \
        << " -l " << predicted_codes[0][9]  /* dilation h */ \
        << " -j " << predicted_codes[0][10] /* dilation w */ \
        << " -p " << predicted_codes[0][11] /* padding h */ \
        << " -q " << predicted_codes[0][12] /* padding w */ \
        << " -g " << "1" /* group */ \
        << " -F " << "1" /* forward conv */ \
        << " -N " << (int) predicted_codes.size() \
        << " -A ";

    int elapsed_time = -1;
    for (int j = 0; j < predicted_codes.size() - 1; j++)
    {
        for (int i = SEP_IDX; i < predicted_codes[0].size(); i++)
        {
            scope << predicted_codes[j][i] << ":";
        }
        scope << elapsed_time << ":"; // unknown elapse time
    }

    for (int i = SEP_IDX; i < predicted_codes[0].size(); i++)
    {
        scope << predicted_codes[predicted_codes.size() - 1][i] << ":";
    }
    scope << elapsed_time ; // unknown elapse time
    

    scope << " >> pred_results.txt" << endl;
    }

}
int verify_prediction(vector<vector<int>> predicted_codes, vector<int> codes, int separation_idx, int no_of_candidates)
{
    for (int k = 0; k < predicted_codes.size(); k++)
    {
        int diff = 0;
        for (int j = SEP_IDX; j < codes.size(); j++)
        {
            //cout << codes[j] << "    " << predicted_codes[j] << endl;
            diff += abs(codes[j] - predicted_codes[k][j]);
        }
        if (diff == 0)
        {
            return(0);
        }
    }
    return(1);
}
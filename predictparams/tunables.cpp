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



bool tunable_is_valid_fwd(vector<int> vec)
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


bool tunable_is_valid_bwd(vector<int> vec)
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

        assert(c % group == 0 && k % group == 0);

        
        int gcd_stride_dilation_h = utility_gcd(stride_h, dilation_h);
        int gcd_stride_dilation_w = utility_gcd(stride_w, dilation_w);

        int y_tilda = stride_h / gcd_stride_dilation_h;
        int x_tilda = stride_w / gcd_stride_dilation_w;

        int y_dot = utility_integer_divide_ceil(y, y_tilda);
        int x_dot = utility_integer_divide_ceil(x, x_tilda);

        int h_tilda = ho + utility_integer_divide_ceil(dilation_h * (y - 1), stride_h);
        int w_tilda = wo + utility_integer_divide_ceil(dilation_w * (x - 1), stride_w);

        int h_tilda_left = utility_integer_divide_floor(
            utility_max(0, pad_h - dilation_h * (y_tilda - 1)), stride_h);
        int w_tilda_left = utility_integer_divide_floor(
            utility_max(0, pad_w - dilation_w * (x_tilda - 1)), stride_w);

        int h_tilda_right = utility_min(
            h_tilda, utility_integer_divide_ceil(pad_h + hi - 1, stride_h) + 1);
        int w_tilda_right = utility_min(
            w_tilda, utility_integer_divide_ceil(pad_w + wi - 1, stride_w) + 1);

        int h_tilda_slice = h_tilda_right - h_tilda_left;
        int w_tilda_slice = w_tilda_right - w_tilda_left;
        int num_of_gemm = y_tilda * x_tilda;

       
        int b = h_tilda_slice * w_tilda_slice;
        b = (nxe == 0) ? (b) : ((b + nxb - 1) / nxb) * nxb;   // pad to nxb modulo when nxe != 0
        int gemm_n = n * b;

        bool unit_conv = (x==1)&&(y==1)&&(stride_h==1)&&(stride_w==1)&&(dilation_h==1)&&(dilation_w==1)&&(pad_h==0)&&(pad_w==0);

        if(gemm_n%gemm_n_per_block!=0){
            // printf("tunable_is_valid false:: gemm_n is %d, gemm_n_per_block is %d, gemm_m is %d, gemm_m_per_block is %d\n", gemm_n,gemm_n_per_block,gemm_m,gemm_m_per_block);
            return false;
        }

        if(gemm_n_per_block%nxb!=0){
            // printf("tunable_is_valid false: gemm_n_per_block%tunable->nxb!=0, gemm_n_per_block is %d, tunable->nxb is %d\n", gemm_n_per_block, tunable->nxb);
            return false;
        }
        //# ho * wo is 4x, gemm_n is 256, hence need batch size 256/4=64x
        if(n%(gemm_n_per_block/nxb)!=0){
            // printf("tunable_is_valid false: n%(gemm_n_per_block/tunable->nxb)!=0, gemm_n_per_block is %d, tunable->nxb is %d\n", gemm_n_per_block, tunable->nxb);
            return false;
        }
        if( (nxe == 0)&& ((h_tilda_slice * w_tilda_slice) % nxb != 0) ){
            return false;
        }
        bool gemm_k_valid = true;
        for(int gemm_id = 0; gemm_id < num_of_gemm; gemm_id++){
            int i_y_tilda = gemm_id / x_tilda;
            int i_x_tilda = gemm_id % x_tilda;
            int y_dot_slice = utility_integer_divide_ceil(y - i_y_tilda, y_tilda);
            int x_dot_slice = utility_integer_divide_ceil(x - i_x_tilda, x_tilda);

            int gemm_k = (k / group) * y_dot_slice * x_dot_slice;
            bool is_gemm_not_empty = gemm_k > 0 && y_dot_slice > 0 && x_dot_slice > 0;
            if(is_gemm_not_empty){
                if(gemm_k % gemm_k_per_block != 0)
                    gemm_k_valid = false;
            }
        }
        if(!gemm_k_valid)
            return false;

        if(nxe == 0 && !unit_conv){
            return false;
        }

        // output vector load limitation, n1b
        if(tensor_b_thread_lengths[3] > 1 && (
            !unit_conv ||
            unit_conv && (ho * wo) % tensor_b_thread_lengths[3] != 0)) {
            return false;
        }

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
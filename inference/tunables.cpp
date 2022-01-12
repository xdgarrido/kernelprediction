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

static inline size_t split_batch_size_x(vector<int> vec, int data_byte)
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
        
        // int data_byte = utility_string_to_data_byte(tunable->precision);
        size_t image_size_input = static_cast<size_t>(c) * hi * wi * data_byte;
        size_t image_size_output = static_cast<size_t>(k) * ho * wo * data_byte;
        size_t size_4g = 0xffffffffUL;
        if (image_size_input >= size_4g || image_size_output >= size_4g)
            return 0;

        size_t image_size = image_size_input >= image_size_output ? image_size_input : image_size_output;
        size_t splited_n = size_4g / image_size;

        // round up splits, we must match
        // 1. splited_n * image_size < size_4g
        // 2. n % splited_n == 0
        // if(splited_n >= n)
        //     return 1;
        assert(splited_n != 0);
        while (splited_n >= 1) {
            // printf("n:%d, splited_n:%d\n", n, splited_n);
            if (n % splited_n == 0 && splited_n * image_size < size_4g)
                break;
            splited_n--;
        }
        assert(splited_n * image_size < size_4g&& n% splited_n == 0);
        return static_cast<size_t>(n) / splited_n;
        
    }

static inline int igemm_get_max_gks_x(int gemm_k, int gemm_k_per_block, int max_log2_splits)
{
    if(gemm_k % gemm_k_per_block != 0)
        return 0;
    int rem = gemm_k / gemm_k_per_block;
    // to find the highest power of 2 value that can divide rem
    // https://www.geeksforgeeks.org/highest-power-of-two-that-divides-a-given-number/
    int rem_pow2 = rem & (~(rem - 1));
    int gks = (int)log2(rem_pow2);
    if(gks > max_log2_splits)
        gks = max_log2_splits;
    return gks;
}

bool tunable_is_valid_fwd(vector<int> vec, string tensor_layout, string precision)
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
    int power_split = vec[43];
    int gemm_k_global_split=0;
    if (power_split == -1)
    {
        gemm_k_global_split = 0;
    }
    else
    {
        gemm_k_global_split = 1 << power_split;
    }
    int merge_e = vec[44];
    int tensor_a_pass_through = vec[45];
    int vector_store = 0;
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

    size_t splits = split_batch_size_x(vec, utility_string_to_data_byte_x(precision));
   // if (splits == 0) {
   //     printf("image size (c*h*w) is bigger than 4g, which is not supported now\n");
   //     return false;
  //  }
    n = n / ((int) splits);   // split batch size here

    int b = ho * wo;
    if (tensor_layout == "nchw")
        b = nxe == 0 ? (ho * wo) : ((ho * wo + nxb - 1) / nxb) * nxb;   // pad to nxb modulo when nxe != 0

    bool unit_conv = (x == 1) && (y == 1) && (stride_h == 1) && (stride_w == 1) && (dilation_h == 1) && (dilation_w == 1) && (pad_h == 0) && (pad_w == 0);

    if (tensor_layout == "nchw") {
        int gemm_m = ((k / group + gemm_m_per_block - 1) / gemm_m_per_block) * gemm_m_per_block;
        int gemm_n = n * b;
        int gemm_k = (c / group) * y * x;

        // support pad to modulo, hence only check when nxe is 0
        if ((gemm_n % gemm_n_per_block != 0) || (gemm_m % gemm_m_per_block != 0))
        {
            return false;
        }

        if (gemm_n_per_block % nxb != 0) {
            //printf("tunable_is_valid false: gemm_n_per_block%tunable->nxb!=0, gemm_n_per_block is %d, tunable->nxb is %d\n", gemm_n_per_block, tunable->nxb);
            return false;
        }

        if (n % (gemm_n_per_block / nxb) != 0) {
            //printf("tunable_is_valid false: n%(gemm_n_per_block/tunable->nxb)!=0, gemm_n_per_block is %d, tunable->nxb is %d\n", gemm_n_per_block, tunable->nxb);
            return false;
        }

        if ((nxe == 0) && ((b % nxb != 0) || (gemm_k % gemm_k_per_block != 0))) {
            return false;
        }

        if ((nxe == 0) && !unit_conv) {
            return false;
        }

        // input vector load limitation, n1b
        if (tensor_b_thread_lengths[3] > 1 && (
            !unit_conv ||
            unit_conv && (hi * wi) % tensor_b_thread_lengths[3] != 0)) {
            return false;
        }

        // weight vector load limitation, c1e
        if (tensor_a_thread_lengths[1] > 1 &&
            gemm_k % tensor_a_thread_lengths[1] != 0) {
            return false;
        }

        // if tb_c1e > 1, only 1x1 case is runable, it can not check gemm_k_padding either.
        if (tensor_b_thread_lengths[1] > 1 && ((x != 1 || y != 1) || (gemm_k % gemm_k_per_block != 0))) {
            return false;
        }

        // if t_c0 > 1, need to check gemmk per block
        if (tensor_b_thread_lengths[0] > 1 && (gemm_k % gemm_k_per_block != 0)) {
            return false;
        }
    }
    else if (tensor_layout == "nhwc") {
        //int gemm_m = n * b;
        // int gemm_n = ((k/group + gemm_n_per_block -1)/gemm_n_per_block) * gemm_n_per_block;
        //int gemm_n = k / group;
        int max_log2_splits = 3;
        //int max_split_num = 0;
        int max_split_num = gemm_k_global_split == 0 ?
                0 : igemm_get_max_gks_x(c / group, gemm_k_per_block, max_log2_splits);

        if ((power_split > -1) && (power_split > max_split_num ))
           return false;

        // support pad to modulo, hence only check when nxe is 0
        //if((gemm_n % gemm_n_per_block != 0) || (gemm_m % gemm_m_per_block != 0))
        //{
        //    return false;
        //}
        if (merge_e) {
            uint32_t s_move_slice_k_y = (gemm_k_per_block / (x * (c / group))) % y;
            uint32_t s_move_slice_k_x = (gemm_k_per_block / (c / group)) % x;
            uint32_t s_move_slice_k_c = gemm_k_per_block % (c / group);
            if ((c / group) >= 0xffffff || y >= 0xffffff || x >= 0xffffff)   // 24 bit
                return false;
            if (s_move_slice_k_y >= 256 || s_move_slice_k_x >= 256 || s_move_slice_k_c >= 256)   // 8 bit
                return false;
        }

        if (tensor_a_thread_lengths[1] == 1 && tensor_b_thread_lengths[1] == 1) {
            ;   // if both 1, indicate padded c support
        }
        else {
            if (((c >> gemm_k_global_split) / group) % gemm_k_per_block != 0)
                return false;
        }

        // if(gemm_m_per_block % tunable->nxb != 0){
        //     //printf("tunable_is_valid false: gemm_n_per_block%tunable->nxb!=0, gemm_n_per_block is %d, tunable->nxb is %d\n", gemm_n_per_block, tunable->nxb);
        //     return false;
        // }

        // if(n % (gemm_m_per_block / tunable->nxb) != 0){
        //     //printf("tunable_is_valid false: n%(gemm_n_per_block/tunable->nxb)!=0, gemm_n_per_block is %d, tunable->nxb is %d\n", gemm_n_per_block, tunable->nxb);
        //     return false;
        // }

        // if((nxe == 0) && ((b % tunable->nxb != 0) || (gemm_k % gemm_k_per_block != 0))){
        //     return false;
        // }

        if ((nxe == 0) && !unit_conv) {
            return false;
        }
        if (precision == "fp16") {
            // fp16 support vector writeout by default. check get_vector_write_out()
            if (tensor_a_thread_lengths[1] == 1 && tensor_b_thread_lengths[1] == 1) {
                ;   // if both 1, k is also write out one by one
            }
            else {
                if (gemm_k_global_split) {
                    if ((k / group) % 2 != 0)
                        return false;
                }
                else {
                    if ((k / group) % utility_gcd_x(gemm_n_per_block, vector_store == 0 ? 8 : vector_store) != 0)
                        return false;
                }
            }
        }

        if (precision == "int8") {
            // fp16 support vector writeout by default. check get_vector_write_out()
            if (tensor_a_thread_lengths[1] == 1 && tensor_b_thread_lengths[1] == 1) {
                ;   // if both 1, k is also write out one by one
            }
            else {
                if (gemm_k_global_split) {
                    assert(false);
                }
                else {
                    if ((k / group) % utility_gcd_x(gemm_n_per_block, vector_store == 0 ? 16 : vector_store) != 0)
                        return false;
                }
            }
        }

        // input vector load limitation, n1b
        //if(tunable->tensor_a_thread_lengths[3] > 1 && (
        //    !unit_conv ||
        //    unit_conv && (hi * wi) % tunable->tensor_a_thread_lengths[3] != 0)) {
        //    return false;
        //}

        // // weight vector load limitation, c1e
        // if(tunable->tensor_a_thread_lengths[1] > 1 &&
        //         gemm_k % tunable->tensor_a_thread_lengths[1] != 0){
        //     return false;
        // }

        // // if tb_c1e > 1, only 1x1 case is runable, it can not check gemm_k_padding either.
        // if(tunable->tensor_b_thread_lengths[1] > 1 && (( x !=1 || y != 1)||(gemm_k % gemm_k_per_block != 0))){
        //     return false;
        // }

        // // if t_c0 > 1, need to check gemmk per block
        // if(tunable->tensor_b_thread_lengths[0] > 1 && (gemm_k % gemm_k_per_block != 0)){
        //     return false;
        // }
    }
    else {
        assert(0);
    }
    return true;

    
}


bool tunable_is_valid_bwd(vector<int> vec, string tensor_layout, string precision)
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
    int power_split = vec[43];
    int gemm_k_global_split=0;
    if (power_split == -1)
    {
        gemm_k_global_split = 0;
    }
    else
    {
        gemm_k_global_split = 1 << power_split;
    }
    int merge_e = vec[44];
    int tensor_a_pass_through = vec[45];
    int vector_store = 0;
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

    size_t splits = split_batch_size_x(vec, utility_string_to_data_byte_x(precision));
    if (splits == 0) {
        printf("image size (c*h*w) is bigger than 4g, which is not supported now\n");
        return false;
    }
    n = n / ((int)splits);   // split batch size here

    int gcd_stride_dilation_h = utility_gcd_x(stride_h, dilation_h);
    int gcd_stride_dilation_w = utility_gcd_x(stride_w, dilation_w);

    int y_tilda = stride_h / gcd_stride_dilation_h;
    int x_tilda = stride_w / gcd_stride_dilation_w;

    int y_dot = utility_integer_divide_ceil_x(y, y_tilda);
    int x_dot = utility_integer_divide_ceil_x(x, x_tilda);

    int h_tilda = ho + utility_integer_divide_ceil_x(dilation_h * (y - 1), stride_h);
    int w_tilda = wo + utility_integer_divide_ceil_x(dilation_w * (x - 1), stride_w);

    int h_tilda_left = utility_integer_divide_floor_x(
            utility_max_x(0, pad_h - dilation_h * (y_tilda - 1)), stride_h);
    int w_tilda_left = utility_integer_divide_floor_x(
            utility_max_x(0, pad_w - dilation_w * (x_tilda - 1)), stride_w);

    int h_tilda_right = utility_min_x(
            h_tilda, utility_integer_divide_ceil_x(pad_h + hi - 1, stride_h) + 1);
    int w_tilda_right = utility_min_x(
            w_tilda, utility_integer_divide_ceil_x(pad_w + wi - 1, stride_w) + 1);

    int h_tilda_slice = h_tilda_right - h_tilda_left;
    int w_tilda_slice = w_tilda_right - w_tilda_left;
    int num_of_gemm = y_tilda * x_tilda;

    bool unit_conv = (x == 1) && (y == 1) && (stride_h == 1) && (stride_w == 1) && (dilation_h == 1) && (dilation_w == 1) && (pad_h == 0) && (pad_w == 0);

    if (tensor_layout == "nchw") {
            
            int b = h_tilda_slice * w_tilda_slice;
            b = (nxe == 0) ? (b) : ((b + nxb - 1) / nxb) * nxb;   // pad to nxb modulo when nxe != 0
            int gemm_n = n * b;
            if (gemm_n % gemm_n_per_block != 0) {
                // printf("tunable_is_valid false:: gemm_n is %d, gemm_n_per_block is %d, gemm_m is %d, gemm_m_per_block is %d\n", gemm_n,gemm_n_per_block,gemm_m,gemm_m_per_block);
                return false;
            }
            if ((tensor_a_thread_lengths[0] != 1 || tensor_a_thread_lengths[1] != 1 ||
                tensor_b_thread_lengths[0] != 1 || tensor_b_thread_lengths[1] != 1) && (k / group) % gemm_k_per_block != 0)
                return false;

            if (gemm_n_per_block % nxb != 0) {
                // printf("tunable_is_valid false: gemm_n_per_block%tunable->nxb!=0, gemm_n_per_block is %d, tunable->nxb is %d\n", gemm_n_per_block, tunable->nxb);
                return false;
            }
            //# ho * wo is 4x, gemm_n is 256, hence need batch size 256/4=64x
            if (n % (gemm_n_per_block / nxb) != 0) {
                // printf("tunable_is_valid false: n%(gemm_n_per_block/tunable->nxb)!=0, gemm_n_per_block is %d, tunable->nxb is %d\n", gemm_n_per_block, tunable->nxb);
                return false;
            }
            if ((nxe == 0) && ((h_tilda_slice * w_tilda_slice) % nxb != 0)) {
                return false;
            }
            bool gemm_k_valid = true;
            for (int gemm_id = 0; gemm_id < num_of_gemm; gemm_id++) {
                int i_y_tilda = gemm_id / x_tilda;
                int i_x_tilda = gemm_id % x_tilda;
                int y_dot_slice = utility_integer_divide_ceil_x(y - i_y_tilda, y_tilda);
                int x_dot_slice = utility_integer_divide_ceil_x(x - i_x_tilda, x_tilda);

                int gemm_k = (k / group) * y_dot_slice * x_dot_slice;
                bool is_gemm_not_empty = gemm_k > 0 && y_dot_slice > 0 && x_dot_slice > 0;
                if (is_gemm_not_empty) {
                    if (gemm_k % gemm_k_per_block != 0)
                        gemm_k_valid = false;
                }
            }
            if (!gemm_k_valid)
                return false;

            if (nxe == 0 && !unit_conv) {
                return false;
            }

            // output vector load limitation, n1b
            if (tensor_b_thread_lengths[3] > 1 && (
                !unit_conv ||
                unit_conv && (ho * wo) % tensor_b_thread_lengths[3] != 0)) {
                return false;
            }
        }
        else if (tensor_layout == "nhwc") {
            int max_log2_splits = 3;
            //int max_split_num = 0;
            int max_split_num = gemm_k_global_split == 0 ?
                0 : igemm_get_max_gks_x(c / group, gemm_k_per_block, max_log2_splits);

            if ((power_split > -1) && (power_split > max_split_num ))
            return false;
            
            if (tensor_a_thread_lengths[1] == 1) {
                ;   // if output k 1, indicate padded k support
            }
            else {
                if (((k >> gemm_k_global_split) / group) % gemm_k_per_block != 0)
                    return false;
            }
            if ((nxe == 0) && !unit_conv) {
                return false;
            }

            if (precision == "fp16") {
                // fp16 support vector writeout by default. check get_vector_write_out()
                if (tensor_a_thread_lengths[1] == 1) {
                    ;   // if output k 1, c is also write out one by one
                }
                else {
                    if (gemm_k_global_split) {
                        if ((c / group) % 2 != 0)
                            return false;
                    }
                    else {
                        if ((c / group) % utility_gcd_x(gemm_n_per_block, vector_store == 0 ? 8 : vector_store) != 0)
                            return false;
                    }
                }
            }

            if (precision == "int8") {
                // fp16 support vector writeout by default. check get_vector_write_out()
                if (tensor_a_thread_lengths[1] == 1) {
                    ;   // if both 1, c is also write out one by one
                }
                else {
                    if (gemm_k_global_split) {
                        assert(false);
                    }
                    else {
                        if ((c / group) % utility_gcd_x(gemm_n_per_block, vector_store == 0 ? 16 : vector_store) != 0)
                            return false;
                    }
                }
            }
        }

        return true;
}

bool tunable_is_valid_wrw(vector<int> vec, string tensor_layout, string precision)
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
    //int nxb = vec[41];
    //int nxe = vec[42];
    
    int grid = vec[43];
    int gemm_k_global_split=0;
    if (grid == -1)
    {
        gemm_k_global_split = 0;
    }
    else
    {
        gemm_k_global_split = grid;
    }
    
    int merge_e = vec[44];
    int tensor_a_pass_through = vec[45];
    int vector_store = 0;
    //int elapsed_time = par[43];
    int tensor_a_thread_lengths[4];
    int tensor_a_cluster_lengths[4];
    int tensor_b_thread_lengths[4];
    int tensor_b_cluster_lengths[4];

    for (int i = 0; i < 4; i++)
    {
        tensor_a_thread_lengths[i] = vec[i + 25];
    }

    for (int i = 0; i < 4; i++)
    {
        tensor_a_cluster_lengths[i] = vec[i + 29];
    }

    for (int i = 0; i < 4; i++)
    {
        tensor_b_thread_lengths[i] = vec[i + 33];
    }

    for (int i = 0; i < 4; i++)
    {
        tensor_b_cluster_lengths[i] = vec[i + 37];
    }

    int nxb = vec[41] == 0 ? 1 : vec[41];
    int b  = vec[42] == 0 ? (ho * wo) : ((ho * wo + nxb - 1) / nxb) * nxb;   // pad to nxb modulo when nxe != 0
    int data_byte = utility_string_to_data_byte_x(precision);


    assert(c % group == 0 && k % group == 0);

    size_t splits = split_batch_size_x(vec, utility_string_to_data_byte_x(precision));
    if (splits == 0) {
        printf("image size (c*h*w) is bigger than 4g, which is not supported now\n");
        return false;
    }
    n = n / ((int)splits);   // split batch size here

    
    int gemmk_blocks = 1 << gemm_k_global_split;
    int n_per_block = n >> gemm_k_global_split;
    int gemm_n = (c / group) * y * x;
    int gemm_k = n * b;

    int nxe = vec[42] == 0 ? 1 : vec[42];
    bool unit_conv = (x == 1) && (y == 1) && (stride_h == 1) && (stride_w == 1) && (dilation_h == 1) && (dilation_w == 1) && (pad_h == 0) && (pad_w == 0);

    if (splits > 1 && gemm_k_global_split == 0)
    {
        // large tensor can only used for gkgs kernel
        return false;
    }

    if (tensor_layout == "nchw") {
        if (((c / group) % (gemm_n_per_block / nxe) != 0) || (((x * y) % nxe) != 0))
        {
            return false;
        }
        if (gemm_k % gemm_k_per_block != 0) {
            //std::cout << __func__ << " false: gemm_n is " << gemm_n << ", gemm_n_per_block is " << gemm_n_per_block << ", gemm_m is " << gemm_m << ", gemm_m_per_block is " << gemm_m_per_block << std::endl;
            return false;
        }

        if (gemm_k_per_block % nxb != 0) {
            //std::cout << __func__ << " false: gemm_n_per_block is " << gemm_n_per_block << ", nxb is " << nxb << std::endl;
            return false;
        }

        int n_n0 = tensor_a_cluster_lengths[0] * tensor_a_thread_lengths[0];

        if (n_n0 > 1) {
            if (n_per_block % (tensor_a_thread_lengths[1] * tensor_a_cluster_lengths[1] * n_n0) != 0) {
                return false;
            }
        }
        else {
            if (n_per_block * b % gemm_k_per_block != 0) {
                return false;
            }
        }

        // input vector load limitation, n1b
        if (tensor_b_thread_lengths[1] > 1 && (
            !unit_conv ||
            unit_conv && (hi * wi) % tensor_b_thread_lengths[1] != 0)) {
            return false;
        }

        // output vector load limitation, n1b
        if (tensor_a_thread_lengths[1] > 1 && (
            !unit_conv ||
            unit_conv && (ho * wo) % tensor_a_thread_lengths[1] != 0)) {
            return false;
        }
        if (b % nxb != 0) {
            //std::cout << __func__ << " false: (ho * wo) is " << (ho * wo) << ", nxb is " << nxb << std::endl;
            return false;
        }
    }
    else {
        if (data_byte == 2) {
            if (c % tensor_b_thread_lengths[3] != 0) {
                return false;
            }
        }
    }

    if ((x * y * stride_h * stride_w != 1) && (vec[42] == 0))
        return false;

    return true;
    
}


void print_batch_file(const std::string& file_name, vector<vector<int>> predicted_codes, string conv_type, string precision, string layout)
{
    igemm_gtc_tunable_t_x* tunable = new (igemm_gtc_tunable_t_x);

    int conv_idx = 1; 

    if (predicted_codes.size() != 0)
    {
    ofstream scope;

    scope.open(file_name, std::ios_base::app);

    if (conv_type == "fwd")
    {
        conv_idx = 1;
    }
    else if (conv_type == "bwd")
    {
        conv_idx = 2;
    }
    else if (conv_type == "wrw")
    {
        conv_idx = 4;
    }
    
    vec2tunable(predicted_codes[0], layout, precision,conv_type,tunable);
    cout << "kernel name :" << encode_kernel_name(tunable) << endl; 

    string fmt(layout);
    transform(fmt.begin(), fmt.end(), fmt.begin(), ::toupper);

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
        << " -F " << conv_idx /* forward conv */ \
        << " -V 0" /*no verification */ \
        << " -i 1" /* iterations*/ \
        << " --in_layout " << fmt \
        << " --fil_layout " << fmt \
        << " --out_layout " << fmt \
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
    delete(tunable);

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


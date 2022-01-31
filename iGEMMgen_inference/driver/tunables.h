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
#include "common.h"
#include "args.h"
#include "utility.h"
#include "igemm_gtc_base.h"
#include "classifier.h"


bool tunable_is_valid_fwd(vector<int> vec, string tensor_layout, string precision);
bool tunable_is_valid_bwd(vector<int> vec, string tensor_layout, string precision);
bool tunable_is_valid_wrw(vector<int> vec, string tensor_layout, string precision);


static inline size_t splitbatchsize(vector<int> vec, int data_byte)
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




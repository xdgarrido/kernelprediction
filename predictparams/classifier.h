#pragma once

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

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include <vector>
#include <tuple>
#include <algorithm>
#include <limits>
#include "parse.h"
#include <immintrin.h>
#include <cmath>
using namespace std;

#define SEP_IDX 15
vector<vector<int>> multiple_predict_parameters(vector<vector<float>> codebook, vector<int> quant_labels, vector<float> normalized_codes, vector<int> codes, int separation_idx, int no_of_candidates);
vector<vector<int>> multiple_predict_parameters_lambdas(vector<vector<float>> codebook, vector<int> quant_labels, vector<float> normalized_codes, vector<int> codes, int separation_idx, vector<vector<float>> lambdas, int no_of_candidates);
vector<vector<int>> multiple_predict_parameters_omegas(vector<vector<float>> codebook, vector<int> quant_labels, vector<float> normalized_codes, vector<int> codes, int separation_idx, vector<vector<float>> omegas, int no_of_candidates);
void print_batch_file(const std::string& file_name, vector<vector<int>> predicted_codes);

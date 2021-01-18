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
using namespace std;

#define SEP_IDX 15

#ifdef __AVXACC__
#ifndef LINUX 
__declspec(align(64)) float ya[8] = { 0,0,0,0,0,0,0,0 };
__declspec(align(64)) float yi[8] = { 0,0,0,0,0,0,0,0 };
#else
 float ya[8] __attribute__((aligned(64)));
 float yi[8] __attribute__((aligned(64)));
#endif 

float hsum_ps_sse3(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);        // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);        // high half -> low half
    sums = _mm_add_ss(sums, shuf);
    return        _mm_cvtss_f32(sums);
}

float hsum256_ps_avx(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
    vlow = _mm_add_ps(vlow, vhigh);             // add the low 128
    return hsum_ps_sse3(vlow);                  // and inline the sse3 version, which is optimal for AVX
    // (no wasted instructions, and all of them are the 4B minimum)
}

inline float compute_distance(const float* p1, const float* p2)
{

    __m256 euc1 = _mm256_setzero_ps();
    const __m256 r1 = _mm256_sub_ps(_mm256_load_ps(&p1[0]), _mm256_load_ps(&p2[0]));
    euc1 = _mm256_fmadd_ps(r1, r1, euc1);

    __m128 vlow = _mm256_castps256_ps128(euc1);
    __m128 vhigh = _mm256_extractf128_ps(euc1, 1); // high 128
    vlow = _mm_add_ps(vlow, vhigh);                // add the low 128

    __m128 shuf = _mm_movehdup_ps(vlow);        // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);          // high half -> low half
    sums = _mm_add_ss(sums, shuf);
    float distance = _mm_cvtss_f32(sums);

    return distance;
}
#endif 

void print_batch_file(const std::string& file_name, vector<vector<int>> predicted_codes)
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
        << " -A ";

    for (int j = 0; j < predicted_codes.size()-1; j++)
    {
        for (int i = SEP_IDX; i < predicted_codes[0].size(); i++)
        {
            scope << predicted_codes[j][i] << ":";
        }
    }

    for (int i = SEP_IDX; i < predicted_codes[0].size() - 1; i++)
    {
        scope << predicted_codes[predicted_codes.size() - 1][i] << ":";
    }

    scope << predicted_codes[predicted_codes.size() - 1][predicted_codes[0].size() - 1];

    scope << " >> pred_results.txt" << endl;
 
}

bool tunable_is_valid(vector<int> vec, vector<int> par)
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
    int tensor_b_thread_lengths[4];


    //printf("%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
    //    n, c, hi, wi, k, y, x, stride_h, stride_w, dilation_h, dilation_w,
    //    pad_h, pad_w, ho, wo);

    for (int i = 0; i < 4; i++)
    {
        tensor_b_thread_lengths[i] = par[i + 33];
    }


    // Print out
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

    int b = nxe == 0 ? (ho * wo) : ((ho * wo + nxb - 1) / nxb) * nxb;   // pad to nxb modulo when nxe != 0

    int gemm_m = k / group;
    int gemm_n = n * b;
    int gemm_k = (c / group) * y * x;

    // support pad to modulo, hence only check when nxe is 0
    if ((gemm_n % gemm_n_per_block != 0) || (gemm_m % gemm_m_per_block != 0) ||
        (gemm_k % gemm_k_per_block != 0))
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

    if ((nxe == 0) && (b % nxb != 0)) {
        return false;
    }

    if (nxe == 0) {
        if ((x != 1) || (y != 1) || (stride_h != 1) || (stride_w != 1) || (dilation_h != 1) || (dilation_w != 1) || (pad_h != 0) || (pad_w != 0)) {
            return false;
        }
    }
    if (tensor_b_thread_lengths[1] > 1 && (x != 1 || y != 1)) {
        return false;
    }
    return true;
}

vector<vector<int>> multiple_predict_parameters(vector<vector<float>> codebook, vector<float> normalized_codes, vector<int> codes, int separation_idx, int no_of_candidates)
{
    int codebook_size = (int)codebook.size();
    int codebook_dim = (int)codebook[0].size();
    int codes_size = (int)codes.size();
    vector <vector<int>> predicted_codes_set;
    vector<tuple<float,int>> dist_table(codebook_size);
    float Delta_alpha_beta = numeric_limits<float>::max();
        
    for (int i = 0; i < codebook_size; i++)
    {
            vector<int> kernel_parameters;
            for (int j = 0; j < codebook_dim; j++)
            {
                kernel_parameters.push_back((int)codebook[i][j]);
            }

            if (tunable_is_valid(codes, kernel_parameters))
            {
#ifndef __AVXACC__
                float dist = 0.;
                for (int j = 0; j < normalized_codes.size()-1; j++)
                {
                    dist += (codebook[i][j] - normalized_codes[j]) * (codebook[i][j] - normalized_codes[j]);
                }
                Delta_alpha_beta = dist;
#else 
                // exclude labels from the distortion calculation
                for (int l = 0; l < normalized_codes.size() - 1; l++)
                {
                    ya[l] = normalized_codes[l];
                    yi[l] = codebook[i][l];
                }
                Delta_alpha_beta = compute_distance(ya, yi);
#endif
            }
            else
            {
                Delta_alpha_beta = numeric_limits<float>::max();
            }
            dist_table[i] = make_tuple(Delta_alpha_beta, i);
    }

    sort(dist_table.begin(), dist_table.end());

    for (int k = 0; k < no_of_candidates; k++)
    {       
            vector<int> predicted_codes;
            // add original codes
            int idx = get<1>(dist_table[k]);
            for (int j = 0; j < separation_idx; j++)
            {
                predicted_codes.push_back((int)codes[j]);
            }

            // add real prediction
            for (int j = separation_idx; j < codebook_dim; j++)
            {
                predicted_codes.push_back((int)codebook[idx][j]);
            }
            predicted_codes_set.push_back(predicted_codes);
    }

    return(predicted_codes_set);
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

vector<float> normalize_codes(vector<int> codes, vector<vector<float>> min_max, bool normalize)
{
    float cmin;
    float cmax;
    float delta;
    float inv_delta;
    int codes_size = (int) codes.size();
    vector<float> ncodes(codes_size);

    for (int i = 0; i < codes_size-1; i++)
    {
        if (normalize)
        {
            cmin = min_max[i][0];
            cmax = min_max[i][1];
            delta = cmax - cmin;

            if (delta != 0.)
            {
                inv_delta = (float)(1. / delta);
            }
            else
            {
                inv_delta = 1.;
            }
            ncodes[i] = inv_delta * ((float)codes[i] - cmin);
        }
        else
        {
            ncodes[i] = (float)codes[i];
        }
    }
    ncodes[codes_size - 1] = (float) codes[codes_size - 1];
    return(ncodes);
}

vector<int> expand_codes(vector<int> codes, string termination, vector<vector<float>> labels)
{
    vector<int> excodes;

    vector<string> result;
    stringstream s_stream(termination); //create string stream from the string
    int count = 0;
    string word;
    while (s_stream.good()) {

        string substr;
        getline(s_stream, substr, ','); //get first string delimited by comma
        count++;
        result.push_back(substr);
    }
    //for (int i = 0; i < result.size(); i++) {    //print all splitted strings
    //    cout << result.at(i) << endl;
    //}
    vector<int> vect;

    for (int i = 0; i < count; i++)
    {
        word = result.at(i);
        vect.push_back(atoi(word.c_str()));
    }
  
    for (int i = 0; i < codes.size() - 1; i++)
    {
        excodes.push_back((int)codes[i]);
    }

    for (int i = 0; i < vect.size(); i++)
    {
        excodes.push_back((int)vect[i]);
    } 
    // these are a by-product of conv width and height [specific to 1x1 conv's]
    excodes.push_back((int)codes[2]);
    excodes.push_back((int)codes[3]);
    
    for (int j = 0; j < labels[0].size(); j++)
    {
        int idx = codes[(int)(codes.size() - 1)];
        excodes.push_back((int)labels[idx][j]);
    }
    return(excodes);
}

vector<float> expand_codebook(vector<float> codes, string termination, vector<vector<float>> labels)
{
    vector<float> excodes;

    vector<string> result;
    stringstream s_stream(termination); //create string stream from the string
    int count = 0;
    string word;
    while (s_stream.good()) {

        string substr;
        getline(s_stream, substr, ','); //get first string delimited by comma
        count++;
        result.push_back(substr);
    }
    //for (int i = 0; i < result.size(); i++) {    //print all splitted strings
    //    cout << result.at(i) << endl;
    //}
    vector<float> vect;

    for (int i = 0; i < count; i++)
    {
        word = result.at(i);
        vect.push_back((float)atof(word.c_str()));
    }

    for (int i = 0; i < codes.size() - 1; i++) // remove label index 
    {
        excodes.push_back(codes[i]);
    }

    for (int i = 0; i < vect.size(); i++)
    {
        excodes.push_back(vect[i]);
    }
    // these are a by-product of conv width and height [specific to 1x1 conv's]
    excodes.push_back(codes[2]);
    excodes.push_back(codes[3]);

   
    for (int j = 0; j < labels[0].size(); j++)
    {
        int idx = (int)codes[(int)(codes.size() - 1)];
        excodes.push_back(labels[idx][j]);
    }
    return(excodes);
}

vector<vector<float>> fread_codes(string codes_set)
{
    vector<vector<float>> codes;
    ifstream scope(codes_set); 
    string line;
    

    if (scope.is_open() == false)
    {
        cout << "Unable to open file:" << codes_set;
        exit(-1);
    }
    vector<string> line_buffer;

    string word;

    while(!scope.eof())
    {
        string substr;
  
        scope >> line;

        vector<string> result;
        stringstream s_stream(line); //create string stream from the string
        int count = 0;
        while (s_stream.good()) {
            
            string substr;
            getline(s_stream, substr, ','); //get first string delimited by comma
            count++;
            result.push_back(substr);
        }
        vector<float> vect;

        for (int i = 0; i < count; i++)
        {
            word = result.at(i);
            vect.push_back((float)atof(word.c_str()));
        }
        // store in ts codebook
        codes.push_back(vect);
    }
    return codes;
}


int main(int argc, char** argv)
{
    FILE* fd_in = NULL;
    Args_t repository, * pArgs;
    char *fname_quant, *fname_cs, *fname_minmax, *fname_labels;
    const string  fname_conv("conv.txt");
    
    
    // parse program input arguments
    pArgs = &repository;
    ParseArgs(argc, argv, pArgs);

 
    fname_quant = pArgs->quant_name;
    fname_minmax = pArgs->minmax_name;
    fname_labels = pArgs->labels_name;
    fname_cs    = pArgs->cs_name;
    int number_of_candidates = pArgs->number_of_candidates; 
    bool normalized_codebook = pArgs->normalized_codebook;
    string pattern(pArgs->pattern);

    string quant_set(fname_quant);
    string cs_set(fname_cs);
    string minmax_set(fname_minmax);
    string labels_set(fname_labels);

    vector<vector<float>> min_max  = fread_codes(minmax_set);
    vector<vector<float>> qs_codes = fread_codes(quant_set);
    vector<vector<float>> cs_codes = fread_codes(cs_set);
    vector<vector<float>> labels   = fread_codes(labels_set);
    vector<vector<float>> exqs_codes;
   
    int count_zeros = 0;
    int count_others = 0;
    
    for (int i = 0; i < qs_codes.size(); i++)
    {
        vector<float> codes;

        for (int j = 0; j < qs_codes[0].size(); j++)
        {
            codes.push_back(qs_codes[i][j]);
        }

         vector<float> codes_complete = expand_codebook(codes, pattern, labels);
         exqs_codes.push_back(codes_complete);
    }

    for (int i = 0; i < cs_codes.size(); i++)
    {
       
        vector<int> codes;
        for (int j = 0; j < cs_codes[0].size(); j++)
        {
            codes.push_back((int)cs_codes[i][j]);
        }
        // normalize conv parameter space

        
        vector<float> ncodes = normalize_codes(codes, min_max, normalized_codebook);
        
        // build a complete conv parameter space
        vector<int> codes_complete = expand_codes(codes, pattern, labels);

        vector<vector<int>> predicted_codes = multiple_predict_parameters(exqs_codes, ncodes, codes_complete, SEP_IDX, number_of_candidates);
        print_batch_file(fname_conv, predicted_codes);

        int val = verify_prediction(predicted_codes, codes_complete, SEP_IDX, number_of_candidates);

        cout << "pred #" << i << " val =" << val << endl;

        if (val == 0)
            count_zeros++;
        else
            count_others++;

    }
    cout << "----------------------------------------" << endl;
    cout << "Results for no_of_candidates = " << number_of_candidates  << endl;
    cout << "Total number of tests = " << cs_codes.size() << endl; 
    cout << "Perfect Matches =" << (100. * (float)count_zeros / (float)cs_codes.size()) << "%" << endl;
    cout << "Others =" << (100. * (float)count_others / (float)cs_codes.size()) << "%" <<  endl;
    cout << "----------------------------------------" << endl;
  
    return 0;
}
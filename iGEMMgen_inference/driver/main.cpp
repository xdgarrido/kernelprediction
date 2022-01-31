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
#include <assert.h>
#include "tunables.h"
#include "preprocessing.h"
int main(int argc, char** argv)
{
    FILE* fd_in = NULL;
    const string  fname_conv("elapsed");
 
    int number_of_candidates = 1; 
    int normalized_codebook = 1;
    int kernel_size = 1;

    
    vector<vector<float>> omegas_fwd, exqs_codes_fwd, norm_fwd, labels_fwd, omegas_bwd, exqs_codes_bwd, norm_bwd, labels_bwd, omegas_wrw, norm_wrw, exqs_codes_wrw, labels_wrw;
    vector<int> quant_labels_fwd, quant_labels_bwd, quant_labels_wrw;
    

    set_up_network(exqs_codes_fwd, quant_labels_fwd, omegas_fwd, norm_fwd, labels_fwd,
        exqs_codes_bwd, quant_labels_bwd, omegas_bwd, norm_bwd, labels_bwd,
        exqs_codes_wrw, quant_labels_wrw, omegas_wrw, norm_wrw, labels_wrw);

    string conv_type("fwd");
    string layout("nhwc");
    string precision("fp32");

    vector<vector<float>> cs_codes = fread_codes("csfwd.csv");
    
   
    int count_zeros = 0;
    int count_others = 0;
    

    double acc_time= 0;
    for (int i = 0; i < cs_codes.size(); i++)
    {
       
        vector<int> codes;
        for (int j = 0; j < cs_codes[0].size(); j++)
        {
            codes.push_back((int)cs_codes[i][j]);
        }
        
        vector<vector<int>> predicted_codes;
        
        clock_t tstart = clock();
        vector<int> codes_complete;
        if (conv_type == "fwd")
        {
            // normalize conv parameter space
            vector<float> ncodes = normalize_codes(codes, norm_fwd, normalized_codebook);
            // build a complete conv parameter space
            codes_complete = expand_codes(codes, kernel_size, labels_fwd);
            predicted_codes = multiple_predict_parameters_omegas(exqs_codes_fwd, quant_labels_fwd, ncodes, codes_complete, SEP_IDX, omegas_fwd, number_of_candidates, conv_type, precision, layout);
        }
        else if (conv_type == "bwd")
        {
            // normalize conv parameter space
            vector<float> ncodes = normalize_codes(codes, norm_bwd, normalized_codebook);
            // build a complete conv parameter space
            codes_complete = expand_codes(codes, kernel_size, labels_bwd);
            predicted_codes = predicted_codes = multiple_predict_parameters_omegas(exqs_codes_bwd, quant_labels_bwd, ncodes, codes_complete, SEP_IDX, omegas_bwd, number_of_candidates, conv_type, precision, layout);
        }
        else
        {
            // normalize conv parameter space
            vector<float> ncodes = normalize_codes(codes, norm_wrw, normalized_codebook);
            // build a complete conv parameter space
            codes_complete = expand_codes(codes, kernel_size, labels_wrw);
            predicted_codes = predicted_codes = multiple_predict_parameters_omegas(exqs_codes_wrw, quant_labels_wrw, ncodes, codes_complete, SEP_IDX, omegas_wrw, number_of_candidates, conv_type, precision, layout);
        }
        clock_t tend = clock();
        double diff = static_cast<double>(tend - tstart);
        acc_time += diff;

        string fname = fname_conv + "_" + conv_type + ".sh";
        print_batch_file(fname, predicted_codes, conv_type, precision, layout);

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
    cout << "Time  (" << "per conv" << ") = " << ( acc_time / ((double)cs_codes.size() * CLOCKS_PER_SEC)) << " secs" <<endl;
    cout << "----------------------------------------" << endl;

    return 0;
}
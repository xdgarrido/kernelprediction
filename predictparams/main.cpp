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
    Args_t repository, * pArgs;
    char *fname_quant, *fname_cs, *fname_norm, *fname_labels, *fname_scales, *convtype, *precisiontype, *layouttype;
    const string  fname_conv("elapsed.sh");
    enum lvq { VQ=1, GRLVQ=2, GMLVQ=3 };
    vector<vector<float>> lambdas;
    vector<vector<float>> omegas;
    
    // parse program input arguments
    pArgs = &repository;
    ParseArgs(argc, argv, pArgs);

 
    fname_quant  = pArgs->quant_name;
    fname_norm   = pArgs->norm_name;
    fname_labels = pArgs->labels_name;
    fname_scales = pArgs->scales_name;
    fname_cs    = pArgs->cs_name;
    convtype = pArgs->conv_type;
    layouttype = pArgs->layout;
    precisiontype = pArgs->precision;

    int number_of_candidates = pArgs->number_of_candidates; 
    int normalized_codebook = pArgs->normalized_codebook;
    int kernel_size = pArgs->kernel_size;

    string quant_set(fname_quant);
    string cs_set(fname_cs);
    string norm_set(fname_norm);
    string labels_set(fname_labels);
    string scales_set(fname_scales);
    string conv_type(convtype);
    string layout(layouttype);
    string precision(precisiontype);

    vector<vector<float>> norm  = fread_codes(norm_set);
    vector<vector<float>> qs_codes = fread_codes(quant_set);
    vector<vector<float>> cs_codes = fread_codes(cs_set);
    vector<vector<float>> labels   = fread_codes(labels_set);
    vector<int> quant_labels;

   

    int distance_type;
    if (scales_set == "none")
    {
        distance_type = VQ;
    }
    else if (scales_set == "lambdas.csv")
    {
        distance_type = GRLVQ;
        lambdas = fread_codes(scales_set);
    }
    else if (scales_set == "omega.csv")
    {
        distance_type = GMLVQ;
        omegas = fread_codes(scales_set);
    }

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
        quant_labels.push_back(qs_codes[i][qs_codes[0].size()-1]);


         vector<float> codes_complete = expand_codebook(codes, kernel_size, labels);
         exqs_codes.push_back(codes_complete);
    }

    double acc_time= 0;
    for (int i = 0; i < cs_codes.size(); i++)
    {
       
        vector<int> codes;
        for (int j = 0; j < cs_codes[0].size(); j++)
        {
            codes.push_back((int)cs_codes[i][j]);
        }
        // normalize conv parameter space

        
        vector<float> ncodes = normalize_codes(codes, norm, normalized_codebook);
        
        // build a complete conv parameter space
        vector<int> codes_complete = expand_codes(codes, kernel_size, labels);
        vector<vector<int>> predicted_codes;
        vector<vector<int>> filtered_predicted_codes;
        
        clock_t tstart = clock();
        switch (distance_type)
        {
        case VQ:
            predicted_codes = multiple_predict_parameters(exqs_codes, quant_labels, ncodes, codes_complete, SEP_IDX, number_of_candidates,conv_type,precision,layout);
            break;
        case GRLVQ:
            predicted_codes = multiple_predict_parameters_lambdas(exqs_codes, quant_labels, ncodes, codes_complete, SEP_IDX, lambdas, number_of_candidates,conv_type,precision,layout);
            break;
        case GMLVQ:
            predicted_codes = multiple_predict_parameters_omegas(exqs_codes, quant_labels, ncodes, codes_complete, SEP_IDX, omegas, number_of_candidates,conv_type,precision, layout);
            break;
        default:
            cout << "invalid selection" << endl;
            break;
        }
        clock_t tend = clock();
        double diff = static_cast<double>(tend - tstart);
        acc_time += diff;
#if 0
        for (int k = 0; k < predicted_codes.size(); k++)
        {
            if (tunable_is_valid(predicted_codes[k]))
            {
                 filtered_predicted_codes.push_back(predicted_codes[k]);
            }
        }
#endif 
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
    cout << "Time  (" << "per conv" << ") = " << ( acc_time / ((double)cs_codes.size() * CLOCKS_PER_SEC)) << " secs" <<endl;
    cout << "----------------------------------------" << endl;
  
    return 0;
}
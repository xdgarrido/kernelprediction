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
#include "preprocessing.h"

static inline size_t conv_out_size_x(size_t in_size, size_t pad, size_t dilation,
    size_t ksize, size_t stride) {
    return (in_size + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1;
}

vector<float> normalize_codes(vector<int> codes, vector<vector<float>> norm, int normalize)
{
    int codes_size = (int) codes.size();
    vector<float> ncodes(codes_size);

    for (int i = 0; i < codes_size-1; i++)
    {
        if (normalize >= 1)
        {
            if (normalize == 1)
            {
                float cmin = norm[i][0];
                float cmax = norm[i][1];
                float delta = cmax - cmin;
                float inv_delta;
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
            else if (normalize == 2)
            {
                float mu = norm[i][0];
                float sigma = norm[i][1];
                float inv_sigma;
                if (sigma != 0.)
                {
                    inv_sigma = (float)(1. / sigma);
                }
                else
                {
                    inv_sigma = 1.;
                }
                ncodes[i] = inv_sigma * ((float)codes[i] - mu);
            }
        }
        else
        {
            ncodes[i] = (float)codes[i];
        }
    }
    ncodes[codes_size - 1] = (float) codes[codes_size - 1];
    return(ncodes);
}

vector<int> expand_codes(vector<int> codes, int kernel_size, vector<vector<float>> labels)
{
    vector<int> excodes;
    // order of features
    // n, c, hi, wi, k, y, x, stride_h, stride_w, dilation_h, dilation_w, pad_h, pad_w, ho, wo
    
    if (kernel_size == 1) // 1by1 kernel
    {
        for (int i = 0; i < codes.size() - 1; i++) // remove label index
        {
            excodes.push_back((int)codes[i]);
        }
        int pattern_array[8] = { 1,1,1,1,1,1,0,0 };
        for (int i = 0; i < 8; i++)
        {
            excodes.push_back((int)pattern_array[i]);
        }
        // these are a by-product of conv height and width [specific to 1x1 conv's]
        excodes.push_back((int)codes[2]);
        excodes.push_back((int)codes[3]);
    }
    else // nbyn kernel 
    {
        int n  = codes[0];
        int c  = codes[1];
        int hi = codes[2];
        int wi = codes[3];
        int k  = codes[4];
        int y = kernel_size;
        int x = kernel_size;
        int stride_h  = codes[6];
        int stride_w = codes[6];
        int dilation_w = codes[7];
        int dilation_h = codes[7];
        int pad_w = codes[8];
        int pad_h = codes[8];
        int ho = (int)conv_out_size_x((size_t)hi, (size_t)pad_h, (size_t)dilation_h, (size_t)y, (size_t)stride_h);
        int wo = (int)conv_out_size_x((size_t)wi, (size_t)pad_w, (size_t)dilation_w, (size_t)x, (size_t)stride_w);
        excodes.push_back(n);
        excodes.push_back(c);
        excodes.push_back(hi);  
        excodes.push_back(wi);
        excodes.push_back(k);
        excodes.push_back(y);
        excodes.push_back(x);
        excodes.push_back(stride_h);   // strides
        excodes.push_back(stride_w);
        excodes.push_back(dilation_h); // dilation
        excodes.push_back(dilation_w);
        excodes.push_back(pad_h);      // pading 
        excodes.push_back(pad_w);
        excodes.push_back(ho);          
        excodes.push_back(wo);

    }
    for (int j = 0; j < labels[0].size(); j++)
    {
        // labels start from 1
        int idx = codes[(int)(codes.size() - 1)];
        excodes.push_back((int)labels[idx][j]);
    }
    return(excodes);
}


vector<float> expand_codebook(vector<float> codes, int kernel_size, vector<vector<float>> labels)
{
    vector<float> excodes;

    if (kernel_size == 1) // 1by1 kernel
    {
        for (int i = 0; i < codes.size() - 1; i++) // remove label index
        {
            excodes.push_back(codes[i]);
        }
        float pattern_array[8] = { 1.,1.,1.,1.,1.,1.,0,0 };
        for (int i = 0; i < 8; i++)
        {
            excodes.push_back(pattern_array[i]);
        }
        // these are a by-product of conv height and width [specific to 1x1 conv's]
        excodes.push_back(codes[2]);
        excodes.push_back(codes[3]);
    }
    else // nbyn kernel 
    {
        float n = codes[0];
        float c = codes[1];
        float hi = codes[2];
        float wi = codes[3];
        float k = codes[4];
        float y = (float) kernel_size;
        float x = (float) kernel_size;
        float stride_h = codes[6];
        float stride_w = codes[6];
        float dilation_w = codes[7];
        float dilation_h = codes[7];
        float pad_w = codes[8];
        float pad_h = codes[8];
        float ho = (float)hi;
        float wo = (float)wi; 
        excodes.push_back(n);
        excodes.push_back(c);
        excodes.push_back(hi);
        excodes.push_back(wi);
        excodes.push_back(k);
        excodes.push_back(y);
        excodes.push_back(x);
        excodes.push_back(stride_h);   // strides
        excodes.push_back(stride_w);
        excodes.push_back(dilation_h); // dilation
        excodes.push_back(dilation_w);
        excodes.push_back(pad_h);      // pading 
        excodes.push_back(pad_w);
        excodes.push_back(ho);
        excodes.push_back(wo);
       
    }


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

void expand_quant(vector<vector<float>> qs_codes, int kernel_size, vector<vector<float>> labels, vector<int> &quant_labels, vector<vector<float>>& exqs_codes)
{

    for (int i = 0; i < qs_codes.size(); i++)
    {
        vector<float> codes;

        for (int j = 0; j < qs_codes[0].size(); j++)
        {
            codes.push_back(qs_codes[i][j]);
        }
        quant_labels.push_back(qs_codes[i][(int)(qs_codes[0].size() - 1)]);


        vector<float> codes_complete = expand_codebook(codes, kernel_size, labels);
        exqs_codes.push_back(codes_complete);
    }

}

void set_up_network(vector<vector<float>>& exqs_codes_fwd, vector<int>& quant_labels_fwd, vector<vector<float>>& omegas_fwd, vector<vector<float>>& norm_fwd, vector<vector<float>>& labels_fwd,
                    vector<vector<float>>& exqs_codes_bwd, vector<int>& quant_labels_bwd, vector<vector<float>>& omegas_bwd, vector<vector<float>>& norm_bwd, vector<vector<float>>& labels_bwd,
                    vector<vector<float>>& exqs_codes_wrw, vector<int>& quant_labels_wrw, vector<vector<float>>& omegas_wrw, vector<vector<float>>& norm_wrw, vector<vector<float>>& labels_wrw)
{
    
    int kernel_size = 1;
    string quant_set_fwd("./heuristics/net80001by1fwd.csv");
    string quant_set_bwd("./heuristics/net80001by1bwd.csv");
    string quant_set_wrw("./heuristics/net80001by1wrw.csv");
    string labels_set_fwd("./heuristics/labels1by1fwd.csv");
    string labels_set_bwd("./heuristics/labels1by1bwd.csv");
    string labels_set_wrw("./heuristics/labels1by1wrw.csv");
    string domain_set_fwd("./heuristics/domain1by1fwd.csv");
    string domain_set_bwd("./heuristics/domain1by1bwd.csv");
    string domain_set_wrw("./heuristics/domain1by1wrw.csv");
    string omega_set_fwd("./heuristics/omega1by1fwd.csv");
    string omega_set_bwd("./heuristics/omega1by1bwd.csv");
    string omega_set_wrw("./heuristics/omega1by1wrw.csv");
    vector<vector<float>> qs_codes_fwd = fread_codes(quant_set_fwd);
    labels_fwd = fread_codes(labels_set_fwd);
    norm_fwd = fread_codes(domain_set_fwd);
    omegas_fwd = fread_codes(omega_set_fwd);
    vector<vector<float>> qs_codes_bwd = fread_codes(quant_set_bwd);
    labels_bwd = fread_codes(labels_set_bwd);
    norm_bwd = fread_codes(domain_set_bwd);
    omegas_bwd = fread_codes(omega_set_bwd);
    vector<vector<float>> qs_codes_wrw = fread_codes(quant_set_wrw);
    labels_wrw = fread_codes(labels_set_wrw);
    norm_wrw = fread_codes(domain_set_wrw);
    omegas_wrw = fread_codes(omega_set_wrw);

    
    expand_quant(qs_codes_fwd, kernel_size, labels_fwd, quant_labels_fwd, exqs_codes_fwd);
    expand_quant(qs_codes_bwd, kernel_size, labels_bwd, quant_labels_bwd, exqs_codes_bwd);
    expand_quant(qs_codes_wrw, kernel_size, labels_wrw, quant_labels_wrw, exqs_codes_wrw);
}
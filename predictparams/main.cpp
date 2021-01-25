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
#include "classifier.h"

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
    char *fname_quant, *fname_cs, *fname_minmax, *fname_labels, *fname_scales;
    const string  fname_conv("conv.txt");
    enum lvq { VQ=1, GRLVQ=2, GMLVQ=3 };
    vector<vector<float>> lambdas;
    vector<vector<float>> omegas;
    
    // parse program input arguments
    pArgs = &repository;
    ParseArgs(argc, argv, pArgs);

 
    fname_quant  = pArgs->quant_name;
    fname_minmax = pArgs->minmax_name;
    fname_labels = pArgs->labels_name;
    fname_scales = pArgs->scales_name;
    fname_cs    = pArgs->cs_name;
    int number_of_candidates = pArgs->number_of_candidates; 
    bool normalized_codebook = pArgs->normalized_codebook;
    string pattern(pArgs->pattern);

    string quant_set(fname_quant);
    string cs_set(fname_cs);
    string minmax_set(fname_minmax);
    string labels_set(fname_labels);
    string scales_set(fname_scales);

    vector<vector<float>> min_max  = fread_codes(minmax_set);
    vector<vector<float>> qs_codes = fread_codes(quant_set);
    vector<vector<float>> cs_codes = fread_codes(cs_set);
    vector<vector<float>> labels   = fread_codes(labels_set);

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

         vector<float> codes_complete = expand_codebook(codes, pattern, labels);
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

        
        vector<float> ncodes = normalize_codes(codes, min_max, normalized_codebook);
        
        // build a complete conv parameter space
        vector<int> codes_complete = expand_codes(codes, pattern, labels);
        vector<vector<int>> predicted_codes;
        
        clock_t tstart = clock();
        switch (distance_type)
        {
        case VQ:
            predicted_codes = multiple_predict_parameters(exqs_codes, ncodes, codes_complete, SEP_IDX, number_of_candidates);
            break;
        case GRLVQ:
            predicted_codes = multiple_predict_parameters_lambdas(exqs_codes, ncodes, codes_complete, SEP_IDX, lambdas, number_of_candidates);
            break;
        case GMLVQ:
            predicted_codes = multiple_predict_parameters_omegas(exqs_codes, ncodes, codes_complete, SEP_IDX, omegas, number_of_candidates);
            break;
        default:
            cout << "invalid selection" << endl;
            break;
        }
        clock_t tend = clock();
        double diff = static_cast<double>(tend - tstart);
        acc_time += diff;

  
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
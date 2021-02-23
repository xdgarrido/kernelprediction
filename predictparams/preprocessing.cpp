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
        // labels start from 1
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

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


#include "parse.h"
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
#include <random>
#include <iterator>
using namespace std;

void print_batch_file(const std::string& file_name, vector<vector<int>> predicted_codes, string conv_type, string precision, string layout)
{
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

        string fmt(layout);
        transform(fmt.begin(), fmt.end(), fmt.begin(), ::toupper);

        for (int i = 0; i < predicted_codes.size(); i++)
        {
            scope << "./out/conv_driver.exe"  \
                << " conv " \
                << " -n " << predicted_codes[i][0]  /* batch */ \
                << " -c " << predicted_codes[i][1]  /* input chanels */ \
                << " -H " << predicted_codes[i][2]  /* height */\
                << " -W " << predicted_codes[i][3]  /* width */ \
                << " -k " << predicted_codes[i][4]  /* output channels */ \
                << " -y " << predicted_codes[i][5]  /* kernel height */ \
                << " -x " << predicted_codes[i][6]  /* kernel width */ \
                << " -u " << predicted_codes[i][7]  /* stride h */ \
                << " -v " << predicted_codes[i][8]  /* stride w */ \
                << " -l " << predicted_codes[i][9]  /* dilation h */ \
                << " -j " << predicted_codes[i][10] /* dilation w */ \
                << " -p " << predicted_codes[i][11] /* padding h */ \
                << " -q " << predicted_codes[i][12] /* padding w */ \
                << " -g " << "1" /* group */ \
                << " -F " << conv_idx /* forward conv */ \
                << " -V 0" /*no verification */ \
                << " -i 5" /* iterations*/ \
                << " --in_layout " << fmt \
                << " --fil_layout " << fmt \
                << " --out_layout " << fmt \
                << " -B conv_data.txt " \
                << "2>&1 | tee >> conv_output.txt" << endl;
        }
    }

}


int main(int argc, char** argv)
{
    FILE* fd_in = NULL, * fd_out = NULL;
    Args_t repository, * pArgs;
    char* fname_in, *fname_out;
    string line, kernel;
    
    // parse program input arguments
    pArgs = &repository;
    ParseArgs(argc, argv, pArgs);

    fname_in    = pArgs->in_name;
    fname_out    = pArgs->out_name;
    int convpar_size = pArgs->convpar_size;
    string precision(pArgs->precision);
    string conv_type(pArgs->conv_type);
    string layout(pArgs->format);

    string scope_set(fname_in);
    string batch_name(fname_out);
    ifstream scope(scope_set);
    
    // check if there is data to be processed

    if (scope.is_open() == false)
    {
        cout << "Unable to open file:" << scope_set;
        exit(-1);
    }

    // read input cvs data
    
    vector<string> line_buffer;
    vector<vector<int>> gs_codes;
   
    while (!scope.eof())
    { 
        string substr;
        scope >> line; 

        vector<string> record;
        stringstream s_stream(line); //create string stream from the string
        while (s_stream.good()) {
            string substr;
            getline(s_stream, substr, ','); //get first string delimited by comma
            record.push_back(substr);
        }
       
        // remove unecessary dimensions or features and copy kernel codes representing classes
        vector<string> compress_record;
        for (int i = 0; i < convpar_size; i++)
        {
                compress_record.push_back(record[i]);
        }

        vector<int> codes;
        for (int i = 0; i < compress_record.size(); i++) 
         {   
              codes.push_back(atoi(compress_record.at(i).c_str()));
         }
         // store in global gs codebook
         gs_codes.push_back(codes);
       
    }
    vector<vector<int>> wanted_codes;
    vector<int> codes;
    vector<int> n{ 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 288, 320, 384, 512, 768, 1024}; // batch size
    vector<int> n_v{ 1, 2, 4, 8, 12, 16 };
    vector<int> c{ 32, 64, 96, 128, 224, 256, 320, 384, 512, 768, 1024, 1536, 1824, 2048, 3072, 4096 }; // input channels
    vector<int> c_v{ 1, 2, 4, 8, 12, 16 };
    vector<int> hi_wi{ 4, 6, 7, 8, 10, 13, 14, 15, 17, 16, 20, 23, 24, 30, 32, 35, 36, 39, 42, 45, 48, 52, 56, 58, 62, 64, 68, 72, 84, 96 ,112, 128 };// height
    vector<int> wi_v{ 176, 256, 352, 426, 512, 640, 854, 1280, 1920, 2560, 3840 };// video width
    vector<int> hi_v{ 144, 240, 288, 256, 360, 480, 512, 720, 1080, 1440, 2160 }; // video height
    vector<int> k{ 32, 64, 96, 128, 224, 256, 320, 384, 512, 768, 1024, 1536, 1824, 2048, 3072, 4096 }; // output channels
    vector<int> k_v{ 1, 2, 4, 8, 12, 16 };
    int y=1;
    int x=1;
    int stride_h=1;
    int stride_w=1;
    int dilation_h=1;
    int dilation_w=1;
    int pad_h=0;
    int pad_w=0;
    //int ho;
    //int wo;
    int group = 1;

    for (int id1 = 0; id1 < n.size(); id1++)
    {
        for (int id2 = 0; id2 < hi_wi.size(); id2++)
        {
            for (int id3 = 0; id3 < c.size(); id3++)
            {
                for (int id4 = 0; id4 < k.size(); id4++)
                {
                    vector<int> codes(convpar_size);
                    codes[0] = n[id1];
                    codes[1] = c[id3];
                    codes[2] = hi_wi[id2];
                    codes[3] = hi_wi[id2];
                    codes[4] = k[id4];
                    codes[5] = y;
                    codes[6] = x;
                    codes[7] = stride_h;
                    codes[8] = stride_w;
                    codes[9] = dilation_h;
                    codes[10]= dilation_w;
                    codes[11]= pad_h;
                    codes[12]= pad_w;
                    codes[13]= hi_wi[id2];
                    codes[14]= hi_wi[id2];
                    wanted_codes.push_back(codes);    
                }
            }
        }
    }

    for (int id1 = 0; id1 < n_v.size(); id1++)
    {
        for (int id2 = 0; id2 < wi_v.size(); id2++)
        {
            for (int id3 = 0; id3 < hi_v.size(); id3++)
            {
                for (int id4 = 0; id4 < c_v.size(); id4++)
                {
                    for (int id5 = 0; id5 < k_v.size(); id5++)
                    {
                        vector<int> codes(convpar_size);
                        codes[0] = n_v[id1];
                        codes[1] = c_v[id4];
                        codes[2] = hi_v[id3];
                        codes[3] = wi_v[id2];
                        codes[4] = k_v[id5];
                        codes[5] = y;
                        codes[6] = x;
                        codes[7] = stride_h;
                        codes[8] = stride_w;
                        codes[9] = dilation_h;
                        codes[10] = dilation_w;
                        codes[11] = pad_h;
                        codes[12] = pad_w;
                        codes[13] = hi_v[id3];
                        codes[14] = wi_v[id2];
                        wanted_codes.push_back(codes);
                    }
                }
            }
        }
    }
    cout << "Size of wanted clusters:" << wanted_codes.size() << endl;
    for (int i = 0; i < gs_codes.size(); i++)
    { 
        for (int j = 0; j < wanted_codes.size(); j++)
        {
            int eq = 0;
            for (int l = 0; l < convpar_size; l++)
            {
                eq += abs(wanted_codes[j][l] - gs_codes[i][l]);
            }
            if (eq == 0)
            {
                wanted_codes.erase(wanted_codes.begin() + j);
                j = 0;
                break;
            }
        }
    }
    cout << "Size of missing clusters after cross checking:" << wanted_codes.size() << endl;
    print_batch_file(batch_name, wanted_codes, conv_type, precision, layout);

}
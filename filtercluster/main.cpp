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

#include "clustering.h"
#include "parse.h"
#include "preprocessing.h"
using namespace std;

vector<int> build_vector(string line)
{
    vector<int> result;
    stringstream s_stream(line); //create string stream from the string
    while (s_stream.good()) {
        string substr;
        getline(s_stream, substr, ','); //get first string delimited by comma
        result.push_back(atoi(substr.c_str()));
    }
    return(result);
}



int main(int argc, char** argv)
{
    FILE* fd_in = NULL, * fd_out = NULL;
    Args_t repository, * pArgs;
    char* fname_in, *fname_ts, *fname_cs, *fname_cs_norm, *fname_lbls, *fname_quant;
    string fname_graph = "rd.txt";
    string fname_hist = "hist.txt";
    string fname_clustering_domain = "domain.csv";
    int clustering_type;
    int test_set_size;
    string line, kernel;
    
    // parse program input arguments
    pArgs = &repository;
    ParseArgs(argc, argv, pArgs);

    fname_in    = pArgs->in_name;
    fname_ts    = pArgs->ts_name;
    fname_cs    = pArgs->cs_name;
    fname_cs_norm    = pArgs->cs_norm_name;
    fname_lbls  = pArgs->lbls_name;
    fname_quant = pArgs->quant_name;

    // codebooks or network sizes to be computed 
    string number_of_clusters_vals(pArgs->number_of_clusters);
    cout << number_of_clusters_vals << endl;
    vector<int> number_of_clusters = build_vector(number_of_clusters_vals);
    
    // fast or slow pnn clustering 
    clustering_type = pArgs->clustering_type;
    
    // percentage of data points kept for testing (outside training set)

    test_set_size = pArgs->test_set_size;
    
    // dimensions removed from input data [begin and end]
    tuple<int, int> removed_dimensions(pArgs->removed_dimensions);
    
    // dimension where starts the attached label
    int label_idx = pArgs->label_idx;

    // normalize the data points
    int normalize_data = pArgs->normalize_data;

    string scope_set(fname_in);
    ifstream scope(scope_set);
    
    // check if there is data to be processed

    if (scope.is_open() == false)
    {
        cout << "Unable to open file:" << scope_set;
        exit(-1);
    }

    // read input cvs data
    
    vector<string> line_buffer;
    vector<vector<int>> gs_codes, labels;
    vector<vector<float>> ts_codes, cs_codes, cs_norm_codes;
   
    while (!scope.eof())
    { 
        string substr;
        scope >> line; 

        vector<string> result;
        stringstream s_stream(line); //create string stream from the string
        while (s_stream.good()) {
            string substr;
            getline(s_stream, substr, ','); //get first string delimited by comma
            result.push_back(substr);
        }
       
        // remove unecessary dimensions or features
        int start = get<0>(removed_dimensions);
        int end   = get<1>(removed_dimensions);

        for (int i = start; i <= end; i++)
        {
            result.erase(result.begin() + start);
        }

       
        vector<int> codes;
        for (int i = 0; i < result.size(); i++) 
         {   
              codes.push_back(atoi(result.at(i).c_str()));
         }
         // store in global gs codebook
         gs_codes.push_back(codes);
       
    }
    vector<float> val1, val2;
    splitgs(gs_codes, ts_codes, cs_codes, cs_norm_codes, label_idx, labels, test_set_size, normalize_data, val1, val2);

    fprint_codes(fname_cs, cs_codes);
    if (normalize_data >= 1)
        fprint_codes(fname_cs_norm, cs_norm_codes);
    fprint_codes(fname_ts, ts_codes);
    print_codes(fname_lbls, labels);

    if (normalize_data >= 1)
    {
        ofstream clustering_domain(fname_clustering_domain);
        clustering_domain.precision(PRECISION_DIGITS);
        for (int i = 0; i < label_idx - 1; i++)
        {
            clustering_domain << val1[i] << "," << val2[i] << endl;
        }
        clustering_domain << val1[label_idx - 1] << "," << val2[label_idx - 1];
    }
    vector<vector<float>>  codebook;
     
    if (clustering_type == 1)
        codebook = fast_cluster_with_distortion((char*)fname_graph.c_str(), ts_codes, number_of_clusters, label_idx, fname_quant);
    else
        codebook = cluster_with_distortion((char*)fname_graph.c_str(), ts_codes, number_of_clusters, label_idx, fname_quant);


    // collect histogram of classes available in the training set
    vector<int> hist_quant = labels_histogram(ts_codes, label_idx, (int)labels.size());

    ofstream hist(fname_hist);
    // record quantizer histogram 

    for (int i=0; i < (int)labels.size()-1; i++)
    {
        hist << hist_quant[i] << endl;
    }
    hist << hist_quant[(int)labels.size() - 1];
   
    return 0;
}
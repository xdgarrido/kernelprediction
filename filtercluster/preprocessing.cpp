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
void find_minmax_normalization(vector<vector<int>> ts, vector<float>& cmin, vector<float>& cmax, vector<float>& inv_cdelta, int label_idx)
{
    vector<int> vec((int)ts.size());
    vector<float>  min_col(label_idx);
    vector<float>  max_col(label_idx);
    vector<float>  idelta_col(label_idx);

    for (int j = 0; j < label_idx; j++)
    {
        for (int i = 0; i < (int)ts.size(); i++)
        {
            vec[i] = ts[i][j];
        }
        int min_val = *min_element(vec.begin(), vec.end());
        int max_val = *max_element(vec.begin(), vec.end());
        int delta = max_val - min_val;
        min_col[j] = (float)min_val;
        max_col[j] = (float)max_val;
        if (min_col[j] == max_col[j])
            idelta_col[j] = 1.f;
        else
            idelta_col[j] = 1.f / ((float)max_val - (float)min_val);

    }


    for (int j = 0; j < label_idx; j++)
    {
        cmin.push_back(min_col[j]);
        cmax.push_back(max_col[j]);
        inv_cdelta.push_back(idelta_col[j]);
    }
}


void remove(std::vector<int>& v)
{
    auto end = v.end();
    for (auto it = v.begin(); it != end; ++it) {
        end = std::remove(it + 1, end, *it);
    }

    v.erase(end, v.end());
}

vector<int> labels_histogram(vector<vector<float>> ts, int idx, int labels_size)
{
    vector<int> labels_hist(labels_size, 0);
    int ts_size = (int)ts.size();

    for (int i = 0; i < ts_size; i++)
    {

        for (int j = 0; j < labels_size; j++)
            if (((int)ts[i][idx]) == j)
                labels_hist[j] ++;

    }
    return(labels_hist);
}

void splitgs(vector<vector<int>> gs, vector<vector<float>>& ts, vector<vector<float>>& cs, vector<vector<float>>& cs_norm, int label_idx, vector<vector<int>>& labels, int test_set_size, bool normalize_data, vector<float>& cmin, vector<float>& cmax)
{
    int gs_size = (int)gs.size();
    int gs_dim = (int)gs[0].size();
    std::random_device rd;
    std::mt19937 g(rd());
    vector<int> v;
    vector<vector<int>> tmp_labels;


    for (int i = 0; i < gs_size; i++)
    {
        vector<int> label;
        for (int j = label_idx; j < gs_dim - 1; j++)
        {
            label.push_back(gs[i][j]);
        }
        tmp_labels.push_back(label);
    }
    int labels_size = (int)tmp_labels.size();
    int labels_dim = (int)tmp_labels[0].size();

    vector<int> replicated_label;
    for (int i = 0; i < labels_size; i++)
    {
        for (int j = i + 1; j < labels_size; j++)
        {
            int diff = 0;
            for (int k = 0; k < labels_dim; k++)
            {
                diff += abs(tmp_labels[i][k] - tmp_labels[j][k]);
            }
            if (diff == 0)
            {
                replicated_label.push_back(j);
            }
        }
    }
    remove(replicated_label);

    for (int i = 0; i < labels_size; i++)
    {
        bool replicated = false;
        for (int j = 0; j < replicated_label.size(); j++)
        {
            if (i == replicated_label[j])
            {
                replicated = true;
            }
        }
        if (!replicated)
        {
            labels.push_back(tmp_labels[i]);
        }
    }

    vector<float> inv_cdelta;

    find_minmax_normalization(gs, cmin, cmax, inv_cdelta, label_idx);


    for (int i = 0; i < gs_size; i++)
    {
        v.push_back(i);
    }

    shuffle(v.begin(), v.end(), g);

    int index = test_set_size;

    for (int i = 0; i < index; i++)
    {
        int id_label = -1;
        for (int j = 0; j < labels.size(); j++)
        {
            int diff = 0;
            vector<float> val;
            for (int k = 0; k < labels_dim; k++)
            {
                diff += (int)abs(gs[v[i]][k + label_idx] - labels[j][k]);
            }
            if (diff == 0)
            {
                id_label = j;
                for (int k = 0; k < label_idx; k++)
                {
                    float c = (float)gs[v[i]][k];
                    val.push_back(c);
                }
                val.push_back((float)id_label);
                cs.push_back(val);
                break;
            }
        }
    }

    if (normalize_data)
    {
        for (int i = 0; i < (int) cs.size(); i++)
        {
            int id_label = -1;
            vector<float> val;
            for (int j = 0; j < label_idx; j++)
            {
            float c = (float)cs[i][j];
            float c_norm = (c - cmin[j]) * inv_cdelta[j];
            val.push_back(c_norm);
            }
            val.push_back((float)cs[i][label_idx]);
            cs_norm.push_back(val);
        }
    }

    for (int i = index; i < gs_size; i++)
    {
        int id_label = -1;
        for (int j = 0; j < labels.size(); j++)
        {
            int diff = 0;
            vector<float> val;
            for (int k = 0; k < labels_dim; k++)
            {
                diff += (int)abs(gs[v[i]][k + label_idx] - labels[j][k]);
            }
            if (diff == 0)
            {
                id_label = j;
                for (int k = 0; k < label_idx; k++)
                {
                    float c = (float)gs[v[i]][k];
                    if (normalize_data)
                    {
                        float c_norm = (c - cmin[k]) * inv_cdelta[k];
                        val.push_back(c_norm);
                    }
                    else
                        val.push_back(c);
                }
                val.push_back((float)id_label);
                ts.push_back(val);
                break;
            }
        }
    }

}



void print_codes(char* codes_file, vector<vector<int>> codes)
{
    ofstream out(codes_file);


    int codebook_size = (int)codes.size();

    if (codebook_size != 0)
    {
        ofstream out(codes_file);
        int no_of_dimensions = (int)codes[0].size();
        for (int i = 0; i < codebook_size - 1; i++)
        {
            for (int j = 0; j < no_of_dimensions - 1; j++)
            {
                int val = codes[i][j];
                out << val << ",";
            }
            int val = (int)codes[i][no_of_dimensions - 1];
            out << val << endl;
        }
        for (int j = 0; j < no_of_dimensions - 1; j++)
        {
            int val = codes[codebook_size - 1][j];
            out << val << ",";
        }
        int val = (int)codes[codebook_size - 1][no_of_dimensions - 1];
        out << val;
        out.close();
    }

}

void fprint_codes(char* codes_file, vector<vector<float>> codes)
{
    ofstream out(codes_file, ofstream::out);
    out.precision(8);

    int codebook_size = (int)codes.size();

    if (codebook_size != 0)
    {
        ofstream out(codes_file);
        int no_of_dimensions = (int)codes[0].size();
        for (int i = 0; i < codebook_size - 1; i++)
        {
            for (int j = 0; j < no_of_dimensions - 1; j++)
            {
                float val = codes[i][j];
                out << val << ",";
            }
            float val = codes[i][no_of_dimensions - 1];
            out << val << endl;
        }
        for (int j = 0; j < no_of_dimensions - 1; j++)
        {
            float val = codes[codebook_size - 1][j];
            out << val << ",";
        }
        float val = codes[codebook_size - 1][no_of_dimensions - 1];
        out << val;
        out.close();
    }

}
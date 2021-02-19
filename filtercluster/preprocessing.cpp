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
#include "parse.h"
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
bool isequalLabel(vector<int> label_a, vector<int> label_b)
{
    int count = 0;
    for (int i = 0; i < label_a.size(); i++)
    {
        if (label_a[i] != label_b[i])
            return (false);
    }
    return(true);
}
int deleteLabelRepeats(vector<vector<int>> &labels)
{
    int posUsed = (int) labels.size();

    for (int i = 0; i < posUsed; ++i)
    {
        int duplicates = 0;
        int j = i + 1;
        // find the first duplicate, if exists
        vector<int> label_a = labels[i];
        for (; j < posUsed; ++j)
        {
            vector<int> label_b = labels[j];
            if (isequalLabel(label_a,label_b)) {
                ++duplicates;
                break;
            }
        }
        // overwrite the duplicated values moving the rest of the elements...
        for (int k = j + 1; k < posUsed; ++k)
        {
            vector<int> label_b = labels[k];
            if (!isequalLabel(label_a, label_b))
            {
                for (int l = 0; l < (int)labels[0].size(); l++)
                    labels[j][l] = labels[k][l];
                ++j;
            }
            // ...but skip other duplicates
            else
            {
                ++duplicates;
            }
        }
        posUsed -= duplicates;
    }
    // clean up (could be limited to the duplicates only)
     labels.erase(labels.begin()+posUsed, labels.end());
    return(posUsed);
}
void splitgs(vector<vector<int>> gs, vector<vector<float>>& ts, vector<vector<float>>& cs, vector<vector<float>>& cs_norm, int label_idx, vector<vector<int>>& labels, int test_set_size, bool normalize_data, vector<float>& cmin, vector<float>& cmax)
{
    int gs_size = (int)gs.size();
    int gs_dim = (int)gs[0].size();
    std::random_device rd;
    std::mt19937 g(rd());
    vector<int> v;
    vector<vector<int>> tmp_labels;
    vector<vector<float>> gs_local;
  


    for (int i = 0; i < gs_size; i++)
    {
        vector<int> label;
        for (int j = label_idx; j < gs_dim - 1; j++)
        {
            label.push_back(gs[i][j]);
        }
        labels.push_back(label);
    }
    deleteLabelRepeats(labels);
    int labels_size = (int)labels.size();
    int labels_dim = (int)labels[0].size();


    vector<float> inv_cdelta;

    find_minmax_normalization(gs, cmin, cmax, inv_cdelta, label_idx);

    for (int i = 0; i < gs_size; i++)
    {
        v.push_back(i);
    }

    shuffle(v.begin(), v.end(), g);
    for (int i = 0; i < gs_size; i++)
    {
        int id_label = -1;
        for (int j = 0; j < labels_size; j++)
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
                gs_local.push_back(val);
                break;
            }
        }
    }


    vector<int> hist = labels_histogram(gs_local, label_idx, labels_size);

    int index = test_set_size;
    int count = 0;
    for (int i = 0; i < gs_size; i++)
    {
        int  label = (int) gs_local[i][label_idx];

        if ((count < index) && (hist[label] > 1))
        {
            cs.push_back(gs_local[i]);
            hist[label] -= 1;
            count++;
        }
        else
        {
            ts.push_back(gs_local[i]);
        }
    }

    if (normalize_data)
    {
        for (int i = 0; i < (int)cs.size(); i++)
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
        for (int i = 0; i < (int)ts.size(); i++)
        {
            for (int j = 0; j < label_idx; j++)
            {
                float c = (float)ts[i][j];
                float c_norm = (c - cmin[j]) * inv_cdelta[j];
                ts[i][j] = c_norm;
            }
        }
        
    }

}



void print_codes(char* codes_file, vector<vector<int>> codes)
{
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
   
    int codebook_size = (int)codes.size();

    if (codebook_size != 0)
    {
        ofstream out(codes_file, ofstream::out);
        out.precision(PRECISION_DIGITS);
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
void fprint_codes_binary(char* codes_file, vector<vector<float>> codes)
{
   
    int codebook_size = (int)codes.size();

    if (codebook_size != 0)
    {
        ofstream out(codes_file, ofstream::binary);
        int no_of_dimensions = (int)codes[0].size();
        for (int i = 0; i < codebook_size; i++)
        {
            for (int j = 0; j < no_of_dimensions; j++)
            {
                float val = codes[i][j];
                out << val;
            }
        }
        out.close();
    }

}
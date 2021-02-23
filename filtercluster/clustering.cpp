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

void print_quant(string codes_file, vector<vector<float>> codes)
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

void print_quant_binary(string codes_file, vector<vector<float>> codes)
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

tuple<int, float, bool> find_nearest_neighbor(vector<vector<float>> y, int a, int separation_idx, vector<float> n_ab, vector<float> n_apb_inv, int Aj)
{
    int no_of_dimensions = (int)y[0].size();
    int y_size = Aj;
    tuple<int, float, bool> q;

    get<0>(q) = 0;
    get<1>(q) = numeric_limits<float>::max();
    get<2>(q) = false;

    float Delta_a_b;
    for (int i = 0; i < y_size; i++)
    {

        if (a == i) continue;
#if 0
        int equals = 0;
        for (int l = separation_idx; l < no_of_dimensions - 1; l++)
        {
            equals += (int)abs(y[a][l] - y[i][l]);
            if (equals != 0) break;
        }
#endif 
        int  equals = (int)abs(y[a][separation_idx] - y[i][separation_idx]);

        if (equals == 0)
        {
#ifndef __AVXACC__       
            float d_a_b = 0;
            float d;
            for (int l = 0; l < separation_idx; l++)
            {
                d = (float)(y[a][l] - y[i][l]);
                d_a_b += d * d;
            }
#else            
            for (int l = 0; l < separation_idx; l++)
            {
                ya[l] = y[a][l];
                yi[l] = y[i][l];
            }
            float d_a_b = compute_distance(ya, yi);
#endif
            int k = (int)(n_ab[a] + n_ab[i]);
            Delta_a_b = (d_a_b * n_ab[a] * n_ab[i]) * n_apb_inv[k];
        }
        else
        {
            // mismatch of labels 
            Delta_a_b = numeric_limits<float>::max();
        }
        if (Delta_a_b < get<1>(q))
        {
            get<0>(q) = i;
            get<1>(q) = Delta_a_b;
        }
    }
    return(q);
}

void merge_vectors(vector<vector<float>>& y, vector<tuple<int, float, bool>>& nn_table, int& a, int& b, int& Aj, int separation_idx, vector<float>& n_ab, vector<float> n_apb_inv)
{
    if (a > b)
    {
        int tmp = b;
        b = a;
        a = tmp;
    }
    int last = Aj - 1;
    // mark all dependent clusters om a and b indexes for recalculation

    for (int j = 0; j <= last; j++)
    {
        //if ((j == a) || (j == b)) continue;
        if ((get<0>(nn_table[j]) == a) || (get<0>(nn_table[j]) == b))
        {
            get<2>(nn_table[j]) = true;
        }
        else
        {
            get<2>(nn_table[j]) = false;
        }
    }
    get<2>(nn_table[a]) = true;

    // calculate new centroid
    int kappa = (int)(n_ab[a] + n_ab[b]);
    for (int l = 0; l < separation_idx; l++)
    {
        float val = n_apb_inv[kappa] * (n_ab[a] * ((float)y[a][l]) + n_ab[b] * ((float)y[b][l]));
        y[a][l] = val;
    }
    // join partitions
    n_ab[a] = (float)kappa;
    int no_of_dimensions = (int)y[0].size();
    // fill empty position
    if (b != last)
    {
        for (int l = 0; l < no_of_dimensions; l++)
        {
            y[b][l] = y[last][l];
        }
        // update index b;
        get<0>(nn_table[b]) = get<0>(nn_table[last]);
        get<1>(nn_table[b]) = get<1>(nn_table[last]);
        get<2>(nn_table[b]) = get<2>(nn_table[last]);
        n_ab[b] = n_ab[last];
        for (int j = 0; j <= last; j++)
        {
            if (get<0>(nn_table[j]) == last)
            {
                get<0>(nn_table[j]) = b;
            }
        }
    }
    // update codebook size
    Aj = Aj - 1;
}
void update_pointers(vector<vector<float>> y, vector<tuple<int, float, bool>>& nn_table, int separation_idx, vector<float>n_ab, vector<float>n_apb_inv, int Aj)
{
    for (int j = 0; j < Aj; j++)
    {
        if (get<2>(nn_table[j]))
        {
            nn_table[j] = find_nearest_neighbor(y, j, separation_idx, n_ab, n_apb_inv, Aj);
            get<2>(nn_table[j]) = false;
        }
    }
}

int find_minimum_distance(vector<vector<float>> y, vector<tuple<int, float, bool>> nn_table, int Aj)
{

    float min_dist = numeric_limits<float>::max();
    int min_index = -1;
    tuple<int, float, bool> q;

    for (int j = 0; j < Aj; j++)
    {
        q = nn_table[j];
        if (get<1>(q) < min_dist)
        {
            min_index = j;
            min_dist = get<1>(q);
        }
    }
    return(min_index);
}

vector<vector<float>> fast_cluster_with_distortion(char* rd_file, vector<vector<float>> ts, vector<int> number_of_clusters, int separation_idx, string quant_fname)
{
    int no_of_dimensions = (int)ts[0].size();
    int ts_size = (int)ts.size();
    vector<tuple<int, float, bool>> nn_table(ts_size);
    vector<float> cmin, cmax;

    // Calculate look-up tables for avoiding divisions
    vector<float> n_apb_inv(ts_size + 1);
    vector<float> n_ab(ts_size);
    for (int i = 1; i <= ts_size; i++)
    {
        n_apb_inv[i] = (float)(1. / ((float)i));
    }
    for (int i = 0; i < ts_size; i++)
    {
        n_ab[i] = 1.f;
    }

    vector<vector<float>> y;
    for (int i = 0; i < ts_size; i++)
    {
        vector<float> vect;
        for (int j = 0; j < no_of_dimensions; j++)
        {
            float val = (float)ts[i][j];
            vect.push_back(val);
        }
        y.push_back(vect);
    }


    // initialize nn_table 

    for (int j = 0; j < ts_size; j++)
    {
        nn_table[j] = find_nearest_neighbor(y, j, separation_idx, n_ab, n_apb_inv, ts_size);
    }

    int j = 0;
    int Aj = ts_size;
    int final_size = number_of_clusters[number_of_clusters.size() - 1];
    int count = 0;
    while (j < (ts_size - final_size))
    {
        int a = find_minimum_distance(y, nn_table, Aj);
        // check if merge is possible
        if (a == -1) break;
        int b = get<0>(nn_table[a]);
        merge_vectors(y, nn_table, a, b, Aj, separation_idx, n_ab, n_apb_inv);
        update_pointers(y, nn_table, separation_idx, n_ab, n_apb_inv, Aj);
        cout << "[iter=" << j << "]";
        j++;
        if (Aj == number_of_clusters[count])
        {
            vector<vector<float>> codebook(Aj);
            for (int i = 0; i < Aj; i++)
            {
                for (int j = 0; j < (int)no_of_dimensions; j++)
                {
                    //codebook[i].push_back((int)round(y[i][j]));
                    codebook[i].push_back(y[i][j]);
                }
            }

            sort(codebook.begin(), codebook.end(),
            [](const std::vector<float>& x, const std::vector<float>& y) 
            { 
               int pivot = (int)(x.size()-1); // label index
               return x[pivot] < y[pivot];
            });
            string file_name(quant_fname + to_string(number_of_clusters[count]) + ".csv");
            print_quant(file_name, codebook);
            count++;
        }

    }

    vector<vector<float>> codebook(Aj);
    for (int i = 0; i < Aj; i++)
    {
        for (int j = 0; j < (int)no_of_dimensions; j++)
        {
            //codebook[i].push_back((int)round(y[i][j]));
            codebook[i].push_back(y[i][j]);
        }
    }

    return(codebook);
}

vector<vector<float>>  cluster_with_distortion(char* rd_file, vector<vector<float>> ts, vector<int> number_of_clusters, int separation_idx, string quant_fname)
{

    ofstream graph(rd_file);

    int no_of_dimensions = (int)ts[0].size();
    int ts_size = (int)ts.size();


    // Calculate look-up table
    vector<float> n_apb_inv(ts_size + 1);
    vector<float> n_ab(ts_size);
    for (int i = 1; i <= ts_size; i++)
    {
        n_apb_inv[i] = (float)(1. / ((float)i));
    }
    for (int i = 0; i < ts_size; i++)
    {
        n_ab[i] = 1.f;
    }

    vector <vector<float>> y;
    for (int i = 0; i < ts_size; i++)
    {
        vector<float> vect;
        for (int j = 0; j < no_of_dimensions; j++)
        {
            float val = ts[i][j];
            vect.push_back(val);
        }
        y.push_back(vect);
    }


    int j = 0;
    int k;
    int Aj = ts_size;
    float Delta_a_b, Delta_alpha_beta;
    int alpha, beta, kappa;
    float d, d_a_b, d_alpha_beta;
    float Dist = 0;
    int old_size = Aj + 1;
    int final_size = number_of_clusters[number_of_clusters.size() - 1];
    int count = 0;
    while ((j < ts_size - final_size) && (old_size == (Aj + 1)))
    {
        Delta_alpha_beta = numeric_limits<float>::max();
        d_alpha_beta = 0;
        old_size = Aj;
        // perform search to find the closest clusters
        bool merge_allowed = false;
        for (int a = 0; a < Aj - 1; a++)
        {
            for (int b = a + 1; b < Aj; b++)
            {
                // check labels if they are the same
#ifndef __AVXACC__
                int equals = 0;
                for (int l = separation_idx; l < no_of_dimensions - 1; l++)
                {
                    equals += (int)abs(y[a][l] - y[b][l]);
                    if (equals != 0) break;
                }
#else
                int  equals = (int)abs(y[a][separation_idx] - y[b][separation_idx]);
#endif
                if (equals == 0)
                {
                    d_a_b = 0;
                    for (int l = 0; l < separation_idx; l++)
                    {
                        d = (float)(y[a][l] - y[b][l]);
                        d_a_b += d * d;
                    }
                    k = (int)(n_ab[a] + n_ab[b]);
                    Delta_a_b = (d_a_b * n_ab[a] * n_ab[b]) * n_apb_inv[k];
                }
                else
                {  // mismatch of labels 
                    Delta_a_b = numeric_limits<float>::max();
                }

                if (Delta_a_b < Delta_alpha_beta) // store the minimum drop of distortion
                {
                    alpha = a;
                    beta = b;
                    kappa = k;
                    Delta_alpha_beta = Delta_a_b;
                    merge_allowed = true;
                }
            }
        }
        // update normalized global distance
        if (merge_allowed)
        {
            if (rd_file != NULL) // get the distortion associated with the set 
                graph << Aj << "  " << Dist << endl;

            Dist += Delta_alpha_beta / (((float)separation_idx * (float)ts_size));
            // evaluate and update new cluster, don't bother to evaluate labels because they are the same
            for (int l = 0; l < separation_idx; l++)
            {
                float val = n_apb_inv[kappa] * (n_ab[alpha] * ((float)y[alpha][l]) + n_ab[beta] * ((float)y[beta][l]));
                y[alpha][l] = val;
            }

            for (int l = 0; l < no_of_dimensions; l++)
            {
                y[beta][l] = y[Aj - 1][l];
            }

            cout << "[iter=" << j << "]";
            n_ab[alpha] = (float)kappa;
            n_ab[beta] = n_ab[Aj - 1];
            Aj--;
            if (Aj == number_of_clusters[count])
            {
                vector<vector<float>> codebook(Aj);
                for (int i = 0; i < Aj; i++)
                {
                    for (int j = 0; j < (int)no_of_dimensions; j++)
                    {
                        //codebook[i].push_back((int)round(y[i][j]));
                        codebook[i].push_back(y[i][j]);
                    }
                }
                string file_name(quant_fname + to_string(number_of_clusters[count]) + ".csv");
                print_quant(file_name, codebook);
                count++;
            }
            j++;
        }
    }
    if (rd_file != NULL)
        graph << Aj << "  " << Dist << endl;

    vector<vector<float>> codebook(Aj);
    for (int i = 0; i < Aj; i++)
    {
        for (int j = 0; j < no_of_dimensions; j++)
        {
            //codebook[i].push_back((int)round(y[i][j]));
            codebook[i].push_back(y[i][j]);
        }
    }

    return(codebook);
}

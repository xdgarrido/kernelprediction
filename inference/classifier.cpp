#include "tunables.h"

#define min(a,b) ((a)<(b)?(a):(b))

#ifdef __AVXACC__
#ifndef LINUX 
__declspec(align(64)) float ya[8] = { 0,0,0,0,0,0,0,0 };
__declspec(align(64)) float yi[8] = { 0,0,0,0,0,0,0,0 };
__declspec(align(64)) float omega[64] = { 0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0 };
__declspec(align(64)) float omega_t[64] = { 0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0 };

#else
float ya[8] __attribute__((aligned(64)));
float yi[8] __attribute__((aligned(64)));
float omega[64]  __attribute__((aligned(64)));
float omega_t[64]  __attribute__((aligned(64)));
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
inline void matvec_8by8(float* A, float* b, float* c, float d[8])
{
    __m256 va, vb, vc, diff, vtemp;
    __m128 vhigh, vresult;

    vb = _mm256_loadu_ps(b);
    vc = _mm256_loadu_ps(c);
    diff = _mm256_sub_ps(vc, vb);

    for (int i = 0; i < 8; i++) {
        va = _mm256_loadu_ps(A + (i * 8)); // matrix_a[i][k]

         // multiply
        vtemp = _mm256_mul_ps(va, diff);

        // add
        // extract higher four floats
        vhigh = _mm256_extractf128_ps(vtemp, 1); // high 128
              // add higher four floats to lower floats
        vresult = _mm_add_ps(_mm256_castps256_ps128(vtemp), vhigh);
        // horizontal add of that result
        vresult = _mm_hadd_ps(vresult, vresult);
        // another horizontal add of that result
        vresult = _mm_hadd_ps(vresult, vresult);
        // store
        d[i] = _mm_cvtss_f32(vresult);
    }
}
#endif 

void remove(vector<tuple<float, int, int> > & v)
{
    
    for (int i = 0; i < v.size(); i++)
    {
        vector<int> equals;
        for (int j = i+1; j < v.size()-1; j++)
        {

            if (get<2>(v[i]) == get<2>(v[j]))
            {
                equals.push_back(j);
            }
        }
        for (int k = 0; k < equals.size(); k++)
        {
            v.erase(v.begin() + equals[k]);

            for (int l = k + 1; l < equals.size(); l++)
            {
                equals[l] -= 1;
            }
            
        }
    }    
}
void adjust(vector<tuple<float, int, int> > & v)
{
    
    for (int i = 0; i < v.size(); i++)
    {
        if (get<2>(v[i]) == -1) continue;
        for (int j = i+1; j < v.size(); j++)
        {

            if (get<2>(v[i]) == get<2>(v[j]))
             {
              get<2>(v[j]) = -1;
            }
        }
    }    
}


vector<float> matvec_mutiply(vector<vector<float>> M, vector<float> x)
{
    vector<float> y(M[0].size(), 0.f);

    for (int i = 0; i < M.size(); i++)
    {
        for (int j = 0; j < M[0].size(); j++)
        {
            y[i] += (M[i][j] * x[j]);
        }
    }
    return(y);
}
vector<vector<int>> multiple_predict_parameters_omegas(vector<vector<float>> codebook, vector<int> quant_labels, vector<float> normalized_codes, vector<int> codes, int separation_idx, vector<vector<float>> omegas, int no_of_candidates, string conv_type,string precision, string layout)
{
    int codebook_size = (int)codebook.size();
    int codebook_dim = (int)codebook[0].size();
    int codes_size = (int)codes.size();
    vector <vector<int>> predicted_codes_set;
    vector<tuple<float, int, int>> dist_table;
    float Delta_alpha_beta = numeric_limits<float>::max();
    vector<vector<float>> omegas_t(omegas.size(), vector<float>(omegas[0].size()));

    // transpose omegas 
#ifndef __AVXACC__
    // Computing transpose of the omegas matrix
    for (int i = 0; i < omegas.size(); ++i)
        for (int j = 0; j < omegas[0].size(); ++j) {
            omegas_t[j][i] = omegas[i][j];
        }
#else
     // Computing transpose of the omegas matrix
    int k = 0;
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) {
            if ((i < omegas.size()) && (j < omegas[0].size()))
            {
                omega[8 * i + j] = omegas[i][j];
                omega_t[8 * j + i] = omegas[i][j];
            }
        }
#endif 
    for (int i = 0; i < codebook_size; i++)
    {
#ifndef __AVXACC__
            vector<float> vec_difference, vec_partialf, vec_partialb;
            for (int j = 0; j < normalized_codes.size() - 1; j++)
            {
                vec_difference.push_back(codebook[i][j] - normalized_codes[j]);
            }
            vec_partialb = matvec_mutiply(omegas, vec_difference);
            vec_partialf = matvec_mutiply(omegas_t, vec_difference);
            float dist = 0.;
            for (int j = 0; j < normalized_codes.size() - 1; j++)
            {
                dist += vec_partialf[j] * vec_partialb[j];
            }
            Delta_alpha_beta = dist;
#else
            for (int l = 0; l < normalized_codes.size() - 1; l++)
            {
                ya[l] = normalized_codes[l];
                yi[l] = codebook[i][l];
            }
            float d1[8], d2[8];
            matvec_8by8(omega, yi, ya, d1);
            matvec_8by8(omega, yi, ya, d2);
            float dist = 0.;
            for (int l = 0; l < normalized_codes.size() - 1; l++)
            {
                dist += d1[l] * d2[l];
            }
            Delta_alpha_beta = dist;
#endif 
        int label = quant_labels[i];
        dist_table.push_back(make_tuple(Delta_alpha_beta, i, label));
    }

    sort(dist_table.begin(), dist_table.end());
    //  remove duplicates in terms of labels
    if (no_of_candidates > 1)
        remove(dist_table);

    int candidates = min(no_of_candidates,(int)dist_table.size());
    for (int k = 0; k < candidates; k++)
    {
        vector<int> predicted_codes;
        // add original codes
        int idx = get<1>(dist_table[k]);
        for (int j = 0; j < separation_idx; j++)
        {
            predicted_codes.push_back((int)codes[j]);
        }

        // add real prediction
        for (int j = separation_idx; j < codebook_dim; j++)
        {
            predicted_codes.push_back((int)codebook[idx][j]);
        }

        if (conv_type == "fwd")
        {
            if (tunable_is_valid_fwd(predicted_codes,layout,precision))
            {
                predicted_codes_set.push_back(predicted_codes);
            }
            else
            {
                // add one more because kernel parameters were not tunable
                if (candidates < dist_table.size() - 1)
                    candidates++;
            }
        }
        else if (conv_type == "bwd")
        {
            if (tunable_is_valid_bwd(predicted_codes,layout,precision))
            {
                predicted_codes_set.push_back(predicted_codes);
            }
            else
            {
                // add one more because kernel parameters were not tunable
                if (candidates < dist_table.size() - 1)
                    candidates++;
            }
        }
        else if (conv_type == "wrw")
        {
            if (tunable_is_valid_wrw(predicted_codes,layout,precision))
            {
                predicted_codes_set.push_back(predicted_codes);
            }
            else
            {
                // add one more because kernel parameters were not tunable
                if (candidates < dist_table.size() - 1)
                    candidates++;
            }
        }
    }

    return(predicted_codes_set);
}

static vector<tuple<float, int, int>> table_dist(10000);
void calcdist(int start_idx, int end_idx, vector<vector<float>> codebook, vector<int> quant_labels, vector<float> normalized_codes) 
{
    for (int i = start_idx; i <= end_idx; i++)
    {
#ifndef __AVXACC__
            vector<float> vec_difference, vec_partialf, vec_partialb;
            for (int j = 0; j < normalized_codes.size() - 1; j++)
            {
                vec_difference.push_back(codebook[i][j] - normalized_codes[j]);
            }
            vec_partialb = matvec_mutiply(omegas, vec_difference);
            vec_partialf = matvec_mutiply(omegas_t, vec_difference);
            float dist = 0.;
            for (int j = 0; j < normalized_codes.size() - 1; j++)
            {
                dist += vec_partialf[j] * vec_partialb[j];
            }
#else
            for (int l = 0; l < normalized_codes.size() - 1; l++)
            {
                ya[l] = normalized_codes[l];
                yi[l] = codebook[i][l];
            }
            float d1[8], d2[8];
            matvec_8by8(omega, yi, ya, d1);
            matvec_8by8(omega, yi, ya, d2);
            float dist = 0.;
            for (int l = 0; l < normalized_codes.size() - 1; l++)
            {
                dist += d1[l] * d2[l];
            }
#endif 
            int label = quant_labels[i];
            table_dist[i] = make_tuple(dist, i, label);
    }

}

/**
 * Spawns n threads
 */
void spawnDistThreads(int n, int codebook_size, vector<vector<float>> codebook, vector<int> quant_labels, vector<float> normalized_codes)
{

    vector<pair<int, int>> idx(n);

    int threadsize = codebook_size / n;

    int i;
    for (i = 0; i < n - 1; i++)
    {
        idx[i].first = i * threadsize;
        idx[i].second = (i + 1) * threadsize - 1;
    }
    idx[n - 1].first = i * threadsize;
    idx[n - 1].second = codebook_size - 1;


    int start_idx;
    int end_idx;
    std::vector<thread> threads(n);
    // spawn n threads:
    for (int i = 0; i < n; i++) {
        start_idx = idx[i].first;
        end_idx = idx[i].second;
        threads[i] = thread(calcdist, start_idx, end_idx, codebook, quant_labels, normalized_codes);
    }

    for (auto& th : threads) {
        th.join();
    }


}

vector<vector<int>>  multiple_predict_parameters_omegas_with_threads(vector<vector<float>> codebook, vector<int> quant_labels, vector<float> normalized_codes, vector<int> codes, int separation_idx, vector<vector<float>> omegas, int no_of_candidates,string conv_type,string precision, string layout)
{
    int codebook_size = (int)codebook.size();
    int codebook_dim = (int)codebook[0].size();
    int codes_size = (int)codes.size();
    vector <vector<int>> predicted_codes_set;
    vector<vector<float>> omegas_t(omegas.size(), vector<float>(omegas[0].size()));

    // transpose omegas 
#ifndef __AVXACC__
    // Computing transpose of the omegas matrix
    for (int i = 0; i < omegas.size(); ++i)
        for (int j = 0; j < omegas[0].size(); ++j) {
            omegas_t[j][i] = omegas[i][j];
        }
#else
     // Computing transpose of the omegas matrix
    int k = 0;
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) {
            if ((i < omegas.size()) && (j < omegas[0].size()))
            {
                omega[8 * i + j] = omegas[i][j];
                omega_t[8 * j + i] = omegas[i][j];
            }
        }
#endif 
    const auto processor_count = std::thread::hardware_concurrency();

    spawnDistThreads(processor_count, codebook_size, codebook, quant_labels, normalized_codes);

    sort(table_dist.begin(), table_dist.end());

    if (no_of_candidates > 1)
        adjust(table_dist);

    int candidates = min(no_of_candidates, (int)table_dist.size());
    for (int k = 0; k < candidates; k++)
    {
        vector<int> predicted_codes;
        // add original codes
        int idx = get<1>(table_dist[k]);
        for (int j = 0; j < separation_idx; j++)
        {
            predicted_codes.push_back((int)codes[j]);
        }

        // add real prediction
        for (int j = separation_idx; j < codebook_dim; j++)
        {
            predicted_codes.push_back((int)codebook[idx][j]);
        }

        if (conv_type == "fwd")
        {
            if (tunable_is_valid_fwd(predicted_codes,layout,precision))
            {
                predicted_codes_set.push_back(predicted_codes);
            }
            else
            {
                // add one more because kernel parameters were not tunable
                if (candidates < table_dist.size() - 1)
                    candidates++;
            }
        }
        else if (conv_type == "bwd")
        {
            if (tunable_is_valid_bwd(predicted_codes,layout,precision))
            {
                predicted_codes_set.push_back(predicted_codes);
            }
            else
            {
                // add one more because kernel parameters were not tunable
                if (candidates < table_dist.size() - 1)
                    candidates++;
            }
        }
        else if (conv_type == "wrw")
        {
            if (tunable_is_valid_wrw(predicted_codes,layout,precision))
            {
                predicted_codes_set.push_back(predicted_codes);
            }
            else
            {
                // add one more because kernel parameters were not tunable
                if (candidates < table_dist.size() - 1)
                    candidates++;
            }
        }
    }

    return(predicted_codes_set);
}



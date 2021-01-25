
#include "classifier.h"


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

void print_batch_file(const std::string& file_name, vector<vector<int>> predicted_codes)
{


    ofstream scope;

    scope.open(file_name, std::ios_base::app);


    scope << "./out/conv_driver.exe"  \
        << " conv " \
        << " -n " << predicted_codes[0][0]  /* batch */ \
        << " -c " << predicted_codes[0][1]  /* input chanels */ \
        << " -H " << predicted_codes[0][2]  /* height */\
        << " -W " << predicted_codes[0][3]  /* width */ \
        << " -k " << predicted_codes[0][4]  /* output channels */ \
        << " -y " << predicted_codes[0][5]  /* kernel height */ \
        << " -x " << predicted_codes[0][6]  /* kernel width */ \
        << " -u " << predicted_codes[0][7]  /* stride h */ \
        << " -v " << predicted_codes[0][8]  /* stride w */ \
        << " -l " << predicted_codes[0][9]  /* dilation h */ \
        << " -j " << predicted_codes[0][10] /* dilation w */ \
        << " -p " << predicted_codes[0][11] /* padding h */ \
        << " -q " << predicted_codes[0][12] /* padding w */ \
        << " -g " << "1" /* group */ \
        << " -F " << "1" /* forward conv */ \
        << " -A ";

    for (int j = 0; j < predicted_codes.size() - 1; j++)
    {
        for (int i = SEP_IDX; i < predicted_codes[0].size(); i++)
        {
            scope << predicted_codes[j][i] << ":";
        }
    }

    for (int i = SEP_IDX; i < predicted_codes[0].size() - 1; i++)
    {
        scope << predicted_codes[predicted_codes.size() - 1][i] << ":";
    }

    scope << predicted_codes[predicted_codes.size() - 1][predicted_codes[0].size() - 1];

    scope << " >> pred_results.txt" << endl;

}

bool tunable_is_valid(vector<int> vec, vector<int> par)
{

    int n = vec[0];
    int c = vec[1];
    int hi = vec[2];
    int wi = vec[3];
    int k = vec[4];
    int y = vec[5];
    int x = vec[6];
    int stride_h = vec[7];
    int stride_w = vec[8];
    int dilation_h = vec[9];
    int dilation_w = vec[10];
    int pad_h = vec[11];
    int pad_w = vec[12];
    int ho = vec[13];
    int wo = vec[14];
    int group = 1;

    int gemm_m_per_block = par[15];
    int gemm_n_per_block = par[16];
    int gemm_k_per_block = par[17];
    int nxb = par[41];
    int nxe = par[42];
    //int elapsed_time = par[43];
    int tensor_b_thread_lengths[4];


    //printf("%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
    //    n, c, hi, wi, k, y, x, stride_h, stride_w, dilation_h, dilation_w,
    //    pad_h, pad_w, ho, wo);

    for (int i = 0; i < 4; i++)
    {
        tensor_b_thread_lengths[i] = par[i + 33];
    }


    // Print out
    // printf(";%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
    //    gemm_m_per_block, gemm_n_per_block, gemm_k_per_block, wave_tile_m, wave_tile_n, wave_tile_k,
    //    wave_step_m, wave_step_n, wave_repeat_m, wave_repeat_n); (15 - 24)

    //for (int i = 0; i < tensor_a_thread_lengths.size(); i++)  (25,26,27,28)
    //{
    //    printf(",%d", tensor_a_thread_lengths[i]);
    //}
    //for (int i = 0; i < tensor_a_cluster_lengths.size(); i++) (29,30,31,32)
    //{
    //    printf(",%d", tensor_a_cluster_lengths[i]);
    //}
    //for (int i = 0; i < tensor_b_thread_lengths.size(); i++)  (33,34,35,36)
    //{
    //    printf(",%d", tensor_b_thread_lengths[i]);
    //}
    //for (int i = 0; i < tensor_b_cluster_lengths.size(); i++)  (37,38,39,40)
    //{
    //    printf(",%d", tensor_b_cluster_lengths[i]);
    //}

    //printf(",%d", nxb); (41)
    //printf(",%d", nxe); (42)
    //printf(",%d\n", (int)(1000 * elapsed_time));

    int b = nxe == 0 ? (ho * wo) : ((ho * wo + nxb - 1) / nxb) * nxb;   // pad to nxb modulo when nxe != 0

    int gemm_m = k / group;
    int gemm_n = n * b;
    int gemm_k = (c / group) * y * x;

    // support pad to modulo, hence only check when nxe is 0
    if ((gemm_n % gemm_n_per_block != 0) || (gemm_m % gemm_m_per_block != 0) ||
        (gemm_k % gemm_k_per_block != 0))
    {
        return false;
    }

    if (gemm_n_per_block % nxb != 0) {
        //printf("tunable_is_valid false: gemm_n_per_block%tunable->nxb!=0, gemm_n_per_block is %d, tunable->nxb is %d\n", gemm_n_per_block, tunable->nxb);
        return false;
    }

    if (n % (gemm_n_per_block / nxb) != 0) {
        //printf("tunable_is_valid false: n%(gemm_n_per_block/tunable->nxb)!=0, gemm_n_per_block is %d, tunable->nxb is %d\n", gemm_n_per_block, tunable->nxb);
        return false;
    }

    if ((nxe == 0) && (b % nxb != 0)) {
        return false;
    }

    if (nxe == 0) {
        if ((x != 1) || (y != 1) || (stride_h != 1) || (stride_w != 1) || (dilation_h != 1) || (dilation_w != 1) || (pad_h != 0) || (pad_w != 0)) {
            return false;
        }
    }
    if (tensor_b_thread_lengths[1] > 1 && (x != 1 || y != 1)) {
        return false;
    }
    return true;
}

vector<vector<int>> multiple_predict_parameters(vector<vector<float>> codebook, vector<float> normalized_codes, vector<int> codes, int separation_idx, int no_of_candidates)
{
    int codebook_size = (int)codebook.size();
    int codebook_dim = (int)codebook[0].size();
    int codes_size = (int)codes.size();
    vector <vector<int>> predicted_codes_set;
    vector<tuple<float, int>> dist_table(codebook_size);
    float Delta_alpha_beta = numeric_limits<float>::max();

    for (int i = 0; i < codebook_size; i++)
    {
        vector<int> kernel_parameters;
        for (int j = 0; j < codebook_dim; j++)
        {
            kernel_parameters.push_back((int)codebook[i][j]);
        }

        if (tunable_is_valid(codes, kernel_parameters))
        {
#ifndef __AVXACC__
            float dist = 0.;
            for (int j = 0; j < normalized_codes.size() - 1; j++)
            {
                dist += (codebook[i][j] - normalized_codes[j]) * (codebook[i][j] - normalized_codes[j]);
            }
            Delta_alpha_beta = dist;
#else 
            // exclude labels from the distortion calculation
            for (int l = 0; l < normalized_codes.size() - 1; l++)
            {
                ya[l] = normalized_codes[l];
                yi[l] = codebook[i][l];
            }
            Delta_alpha_beta = compute_distance(ya, yi);
#endif
        }
        else
        {
            Delta_alpha_beta = numeric_limits<float>::max();
        }
        dist_table[i] = make_tuple(Delta_alpha_beta, i);
    }

    sort(dist_table.begin(), dist_table.end());

    for (int k = 0; k < no_of_candidates; k++)
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
        predicted_codes_set.push_back(predicted_codes);
    }

    return(predicted_codes_set);
}

vector<vector<int>> multiple_predict_parameters_lambdas(vector<vector<float>> codebook, vector<float> normalized_codes, vector<int> codes, int separation_idx, vector<vector<float>> lambdas, int no_of_candidates)
{
    int codebook_size = (int)codebook.size();
    int codebook_dim = (int)codebook[0].size();
    int codes_size = (int)codes.size();
    vector <vector<int>> predicted_codes_set;
    vector<tuple<float, int>> dist_table(codebook_size);
    float Delta_alpha_beta = numeric_limits<float>::max();
    vector<float> lambda_square_root(lambdas[0].size());
    for (int i = 0; i < lambdas[0].size(); i++)
    {
        lambda_square_root[i] = sqrt(lambdas[0][i]);
    }

    for (int i = 0; i < codebook_size; i++)
    {
        vector<int> kernel_parameters;
        for (int j = 0; j < codebook_dim; j++)
        {
            kernel_parameters.push_back((int)codebook[i][j]);
        }

        if (tunable_is_valid(codes, kernel_parameters))
        {
#ifndef __AVXACC__
            float dist = 0.;
            for (int j = 0; j < normalized_codes.size() - 1; j++)
            {
                dist += lambdas[0][j] * (codebook[i][j] - normalized_codes[j]) * (codebook[i][j] - normalized_codes[j]);
            }
            Delta_alpha_beta = dist;
#else 
            // exclude labels from the distortion calculation
            for (int l = 0; l < normalized_codes.size() - 1; l++)
            {
                ya[l] = lambda_square_root[l] * normalized_codes[l];
                yi[l] = lambda_square_root[l] * codebook[i][l];
            }
            Delta_alpha_beta = compute_distance(ya, yi);
#endif
        }
        else
        {
            Delta_alpha_beta = numeric_limits<float>::max();
        }
        dist_table[i] = make_tuple(Delta_alpha_beta, i);
    }

    sort(dist_table.begin(), dist_table.end());

    for (int k = 0; k < no_of_candidates; k++)
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
        predicted_codes_set.push_back(predicted_codes);
    }

    return(predicted_codes_set);
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
vector<vector<int>> multiple_predict_parameters_omegas(vector<vector<float>> codebook, vector<float> normalized_codes, vector<int> codes, int separation_idx, vector<vector<float>> omegas, int no_of_candidates)
{
    int codebook_size = (int)codebook.size();
    int codebook_dim = (int)codebook[0].size();
    int codes_size = (int)codes.size();
    vector <vector<int>> predicted_codes_set;
    vector<tuple<float, int>> dist_table(codebook_size);
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
        vector<int> kernel_parameters;
        for (int j = 0; j < codebook_dim; j++)
        {
            kernel_parameters.push_back((int)codebook[i][j]);
        }

        if (tunable_is_valid(codes, kernel_parameters))
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
        }
        else
        {
            Delta_alpha_beta = numeric_limits<float>::max();
        }
        dist_table[i] = make_tuple(Delta_alpha_beta, i);
    }

    sort(dist_table.begin(), dist_table.end());

    for (int k = 0; k < no_of_candidates; k++)
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
        predicted_codes_set.push_back(predicted_codes);
    }

    return(predicted_codes_set);
}


#include "learning.h"


#define SEP_IDX 15

#ifdef __AVXACC__
#ifndef LINUX 
__declspec(align(64)) float y_a[8] = { 0,0,0,0,0,0,0,0 };
__declspec(align(64)) float y_b[8] = { 0,0,0,0,0,0,0,0 };
#else
float y_a[8] __attribute__((aligned(64)));
float y_b[8] __attribute__((aligned(64)));
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
int verify_prediction(vector<int> predicted_labels, int label)
{
    for (int k = 0; k < (int)predicted_labels.size(); k++)
    {
        int diff = 0;
        diff = abs(label - predicted_labels[k]);

        if (diff == 0)
        {
            return(0);
        }
    }
    return(1);
}

float calculate_accuracy(vector<vector<float>> codebook, vector<vector<float>> cs, int  number_of_candidates)
{

    int codebook_size = (int)codebook.size();
    int codebook_dim = (int)codebook[0].size();
    int codes_size = (int)cs.size();
    vector<tuple<float, int>> dist_table(codebook_size);
    float Delta_alpha_beta = numeric_limits<float>::max();
    int count_zeros = 0;
    int count_others = 0;


    for (int h = 0; h < codes_size; h++)
    {

        vector<float> codes = cs[h];

        for (int i = 0; i < codebook_size; i++)
        {

#ifndef __AVXACC__
            float dist = 0.;
            for (int j = 0; j < codebook_dim - 1; j++)
            {
                dist += (codebook[i][j] - codes[j]) * (codebook[i][j] - codes[j]);
            }
            Delta_alpha_beta = dist;
#else 
            // exclude labels from the distortion calculation
            for (int j = 0; j < codebook_dim - 1; j++)
            {
                y_a[j] = codes[j];
                y_b[j] = codebook[i][j];
            }
            Delta_alpha_beta = compute_distance(y_a, y_b);
#endif

            dist_table[i] = make_tuple(Delta_alpha_beta, i);
        }

        sort(dist_table.begin(), dist_table.end());

        vector<int> predicted_labels;
        for (int k = 0; k < number_of_candidates; k++)
        {
            int idx = get<1>(dist_table[k]);
            predicted_labels.push_back((int)codebook[idx][codebook_dim - 1]);
        }

        //cout << predicted_labels[0] << ' ' << (int)codes[codebook_dim - 1];
        int val = verify_prediction(predicted_labels, (int)codes[codebook_dim - 1]);

        //cout << "pred #" << h << " val =" << val << endl;

        if (val == 0)
            count_zeros++;
        else
            count_others++;
    }

    float accuracy = (float)(100. * (float)count_zeros / (float)codes_size);
    return (accuracy);
}

float  sgd(float mu, float beta)
{
    float tmp = 1.f / (1.f + exp(-mu * beta));
    return(tmp);
}

float  sgd_prime(float mu, float beta)
{
    float tmp = 1.f / (1.f + (float)exp(-mu * beta));
    float tmp_prime = beta * tmp * (1.f - tmp);
    return(tmp_prime);
}

float  swish(float mu, float beta)
{
    float tmp = mu / (1.f + exp(-mu * beta));
    return(tmp);
}

float  swish_prime(float mu, float beta)
{
    float tmp = mu / (1.f + (float)exp(-mu * beta));
    float sgd = 1.f / (1.f + (float)exp(-mu * beta));
    float tmp_prime = beta * tmp + sgd * (1.f - beta * tmp);
    return(tmp_prime);
}

float calc_cost(vector<float> cost_vector)
{
    float acc = 0;
    for (int i = 0; i < (int)cost_vector.size(); i++)
    {
        acc += cost_vector[i];
    }
    return(acc);
}

float step_based_learning(float learning_rate, float decay, int drop_rate, int n)
{
    float tmp = floor((float)(1 + n) / (float)drop_rate);
    float learning = learning_rate * pow(decay, tmp);
    return(learning);
}

float exp_learning(float learning_rate_start, float learning_rate_end, int epochs, int e)
{
    float tmp = learning_rate_end / learning_rate_start;
    float exponent = ((float) e) / ((float)epochs);
    float learning = learning_rate_start * pow(tmp,exponent);
    return(learning);
}

int glvq(vector<vector<float>>& codebook, vector<vector<float>> ts, vector<vector<float>> cs, LearningArgs_t params, int number_of_candidates)
{
    int codebook_size = (int)codebook.size();
    int dim = (int)codebook[0].size();
    int dim_1 = dim - 1;
    int ts_size = (int)ts.size();
    float Delta_alpha_beta;
    std::random_device rd;
    std::mt19937 g(rd());
    vector<int> v;
    float cost = (float)ts_size;
    vector<float> cost_vector(ts_size);
    fill(cost_vector.begin(), cost_vector.end(), 1.f);

    // set learning parameters
    float learning_rate_start = params.learning_rate_start;
    float learning_rate_end = params.learning_rate_end;
    float learning_decay = params.decay;
    int checkpoint = params.checkpoint;
    int learning_drop_rate = params.drop_rate;
    int epochs = params.epochs; 
    int final_iter = (int)(epochs * ts_size);
    float learning;

    for (int i = 0; i < ts_size; i++)
    {
        v.push_back(i);
    }

    int t = 0;
    for (int e = 0; e < epochs; e++)
    {  
        shuffle(v.begin(), v.end(), g);;
        
        for (int i = 0; i < ts_size; i++)
        {
            if (params.learning_function == "step_based_learning")
                learning = step_based_learning(learning_rate_start, learning_decay, learning_drop_rate, t);
            else if (params.learning_function == "exp_learning")
                learning = exp_learning(learning_rate_start,learning_rate_end,epochs,e);
            else
            { 
                cout << "learning method not found!" << endl;
                return(-1);
            }


            vector<pair<float, int> > vp;
            vector <float> ts_vector = ts[v[i]];

            for (int j = 0; j < codebook_size; j++)
            {

                // same class
#ifndef __AVXACC__
                float dist = 0.;
                for (int k = 0; k < dim_1; k++)
                {
                    dist += (codebook[j][k] - ts_vector[k]) * (codebook[j][k] - ts_vector[k]);
                }
                Delta_alpha_beta = dist;
#else 
                    // exclude labels from the distortion calculation
                for (int k = 0; k < dim_1; k++)
                {
                    y_a[k] = ts_vector[k];
                    y_b[k] = codebook[j][k];
                }
                Delta_alpha_beta = compute_distance(y_a, y_b);
#endif
                vp.push_back(make_pair(Delta_alpha_beta, j));

            }
            sort(vp.begin(), vp.end());

            int datalabel = (int)ts_vector[dim_1];
            int count = 0;
            int idxa = vp[count].second;
            while (idxa != datalabel)
            {
                count++;
                idxa = vp[count].second;
            }
            float da = vp[count].first;

            count = 0;
            int idxb = vp[count].second;
            while (idxb == datalabel)
            {
                count++;
                idxb = vp[count].second;
            }
            float db = vp[count].first;

            float mu = ((da - db) / (da + db));
            float gab = 4.f / ((da + db) * (da + db));
            gab = gab * swish_prime(mu,2.0);

            for (int k = 0; k < dim_1; k++)
            {
              float delta_a = learning * gab * db * (ts_vector[k] - codebook[idxa][k]);
              float delta_b = learning * gab * da * (ts_vector[k] - codebook[idxb][k]);
              codebook[idxa][k] += delta_a;
              codebook[idxb][k] -= delta_b;
            }
            cost_vector[i] = swish(mu,2.);

            //cout << "[iter=" << iter << "]" << '%t';
            if ((t % checkpoint) == 0)
            {
                cout << "[iter=" << t << "]";
                cout << "[epoch=" << e << "]";
                cout << endl << "learning_rate =" << learning << endl;
                float accuracy = calculate_accuracy(codebook, ts, number_of_candidates);
                cout << "ts accuracy  =" << accuracy << endl;
                cost = calc_cost(cost_vector);
                cout << "cost =" << cost << endl;

            }
            t++;
        }
    }
return(0);
}


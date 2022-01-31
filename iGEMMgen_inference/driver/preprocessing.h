#include "classifier.h"
using namespace std;
vector<float> normalize_codes(vector<int> codes, vector<vector<float>> norm, int normalize);
vector<int> expand_codes(vector<int> codes, int kernel_size, vector<vector<float>> labels);
vector<float> expand_codebook(vector<float> codes, int kernel_size, vector<vector<float>> labels);
void expand_quant(vector<vector<float>> qs_codes, int kernel_size, vector<vector<float>> labels, vector<int>& quant_labels, vector<vector<float>>& exqs_codes);
vector<vector<float>> fread_codes(string codes_set);

void set_up_network(vector<vector<float>>& exqs_codes_fwd, vector<int>& quant_labels_fwd, vector<vector<float>>& omegas_fwd, vector<vector<float>>& norm_fwd, vector<vector<float>>& labels_fwd,
    vector<vector<float>>& exqs_codes_bwd, vector<int>& quant_labels_bwd, vector<vector<float>>& omegas_bwd, vector<vector<float>>& norm_bwd, vector<vector<float>>& labels_bwd,
    vector<vector<float>>& exqs_codes_wrw, vector<int>& quant_labels_wrw, vector<vector<float>>& omegas_wrw, vector<vector<float>>& norm_wrw, vector<vector<float>>& labels_wrw);

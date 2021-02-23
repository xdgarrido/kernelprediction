#include "classifier.h"
vector<float> normalize_codes(vector<int> codes, vector<vector<float>> norm, int normalize);
vector<int> expand_codes(vector<int> codes, string termination, vector<vector<float>> labels);
vector<float> expand_codebook(vector<float> codes, string termination, vector<vector<float>> labels);
vector<vector<float>> fread_codes(string codes_set);

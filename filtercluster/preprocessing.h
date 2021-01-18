#pragma once
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
void splitgs(vector<vector<int>> gs, vector<vector<float>>& ts, vector<vector<float>>& cs, vector<vector<float>>& cs_norm, int label_idx, vector<vector<int>>& labels, int test_set_size, bool normalize_data, vector<float>& cmin, vector<float>& cmax);
vector<int> labels_histogram(vector<vector<float>> ts, int idx, int labels_size);
void  print_codes(char* codes_file, vector<vector<int>> codes);
void fprint_codes(char* codes_file, vector<vector<float>> codes);
void fprint_codes_binary(char* codes_file, vector<vector<float>> codes);
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
#include <immintrin.h>
#include <cfloat>
using namespace std;
vector<vector<float>> fast_cluster_with_distortion(char* rd_file, vector<vector<float>> ts, vector<int> number_of_clusters, int separation_idx, string quant_fname, int labels_size);
vector<vector<float>>  cluster_with_distortion(char* rd_file, vector<vector<float>> ts, vector<int> number_of_clusters, int separation_idx, string quant_fname, int labels_size);
void gla(vector<vector<float>> ts, vector<vector<float>>& codebook, float epsilon, int number_of_iterations);
void print_codes(char* codes_file, vector<vector<int>> codes);

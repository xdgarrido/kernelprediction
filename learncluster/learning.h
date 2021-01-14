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
#include <utility>
#include <math.h>
#include "parse.h"
#include <immintrin.h>

using namespace std;
void glvq(vector<vector<float>>& codebook, vector<vector<float>> ts, vector<vector<float>> cs, float window_distance, float learning_rate, float decay, int drop_rate, int epochs, int checkpoint, int number_of_candidates);



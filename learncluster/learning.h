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
#include <random>
#include "parse.h"
#include <immintrin.h>

using namespace std;
typedef struct LearningParams {
	float learning_rate_start;
	float learning_rate_end;
	float decay; 
	int drop_rate;
	int epochs;
	int checkpoint;
	string learning_function; 
} LearningArgs_t;

int glvq(vector<vector<float>>& codebook, vector<vector<float>> ts, vector<vector<float>> cs, LearningArgs_t params, int number_of_candidates);



#include<iostream>
#include <tuple>
#include <algorithm>
typedef struct Inputs {
	char *in_name;
	char *ts_name;
	char *cs_name;
	char *cs_norm_name;
	char *lbls_name;
	char *quant_name;
	char *number_of_clusters;
	int  clustering_type;
	std::tuple<int, int> removed_dimensions;
	int  label_idx;
	bool verbose;
	bool no_clustering;
	bool normalize_data;
	int test_set_size;
} Args_t;
void ParseArgs(int argc, char *argv[], Args_t *p);
#define PRECISION_DIGITS 10

#include<iostream>
#include <tuple>
#include <algorithm>
#define RECORD_LENGTH 47
typedef struct Inputs {
	char *in_name;
	char *ts_name;
	char *cs_name;
	char *cs_norm_name;
	char *lbls_name;
	char *quant_name;
	char *number_of_clusters;
	// pattern idx has 46 records 
	// 0...14 = features; 
	// 15...45 = kernel parameters; 
	// 46 = elapsed time in microsecs
	int  pattern_idx[RECORD_LENGTH];
	int  clustering_type;
	int  label_idx;
	bool verbose;
	bool no_clustering;
	int normalize_data;
	int test_set_size;
} Args_t;
void ParseArgs(int argc, char *argv[], Args_t *p);
#define PRECISION_DIGITS 16

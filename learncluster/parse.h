#include<iostream>
#include<string>
typedef struct Inputs {
	char *quant_name;
	char *minmax_name;
	char *cs_name;
	char *ts_name;
	char *learning_function;
	int number_of_candidates;
	float window_distance;
	float learning_rate_start;
	float learning_rate_end;
	float decay; 
	int drop_rate;
	int epochs;
	int checkpoint; 
	bool verbose;
} Args_t;


void ParseArgs(int argc, char *argv[], Args_t *p);

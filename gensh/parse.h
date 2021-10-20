#include<iostream>
#include <tuple>
#include <algorithm>
#define RECORD_LENGTH 47
typedef struct Inputs {
	char *in_name;
	char *out_name;
	char *format;
	char *precision;
	char *conv_type;
	int convpar_size;

} Args_t;
void ParseArgs(int argc, char *argv[], Args_t *p);
#define PRECISION_DIGITS 16

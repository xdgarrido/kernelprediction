/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "parse.h"
using namespace std;
void Usage(char *program_name)
{
	cout << "Usage is %s [options]" << program_name << endl;
	cout << "Options:" << endl;
	cout << "  -i convolution parameters file name (ex: cs.csv)" << endl;
	cout << "  -g quantizer file name (ex: quant6000.csv)" << endl;
	cout << "  -l labels file name (ex: labels.csv)" << endl;
	cout << "  -m normaization scales file name (ex: domain.csv)" << endl;
	cout << "  -c 0: do not apply normalization 1: apply min-max normalization 2: apply z-score normalization" << endl;
	cout << "  -n number of predictors" << endl;
	cout << "  -v verbose" << endl;
	cout << "  -s classifier used (ex: none (euclidian distance, lambdas.csv (weighted euclidian distance), omega.csv (mahalanobis distance) " << endl;
	cout << "  -c normalized_classifier (1: it is, 0:it isn't)"  << endl;
	exit(1);
}


void ParseArgs(int argc, char *argv[], Args_t *p)
{
	char *program_name = NULL;
	char *quant_name=NULL;
	char *cs_name = NULL;
	char *labels_name = NULL;
	char *norm_name = NULL;
	char *scales_name = NULL;
	char *pattern = NULL;
	// default names
	const char* cs_array      = "cs.csv";
	const char* minmax_array  = "domain.csv";
	const char* lbls_array    = "labels.csv";
	const char* scales_array  = "none";
	const char* pattern_array = "1,1,1,1,1,1,0,0";
	int verbose = 0; 
	int normalized_codebook = 1; // min-max as default
	bool error = true;
	int number_of_candidates = 1;
	char nchar = 2;

	program_name = argv[0];
	while ((argc > 1) && (argv[1][0] == '-'))
	{
		// nchar denotes the first valid letter after '-option'
		nchar = 2;
		// argv[1][1] is the actual option character
		switch (argv[1][1])
		{

		case 'i':
			while (argv[1][nchar] == '\0')
				nchar++;
			cs_name = &argv[1][nchar];
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;
		case 'g':
			while (argv[1][nchar] == '\0')
				nchar++;
			quant_name = &argv[1][nchar];
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;
		case 'l':
			while (argv[1][nchar] == '\0')
				nchar++;
			labels_name = &argv[1][nchar];
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;
		case 'm':
			while (argv[1][nchar] == '\0')
				nchar++;
			norm_name = &argv[1][nchar];
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;
		case 'p':
			while (argv[1][nchar] == '\0')
				nchar++;
			pattern = &argv[1][nchar];
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;
		case 'n':
			while (argv[1][nchar] == '\0')
				nchar++;
			number_of_candidates = atoi(&argv[1][nchar]);
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;
		case 'v':
			while (argv[1][nchar] == '\0')
				nchar++;
			verbose = atoi(&argv[1][nchar]);
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;
		case 'c':
			while (argv[1][nchar] == '\0')
				nchar++;
			normalized_codebook = atoi(&argv[1][nchar]);
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;
		case 's':
			while (argv[1][nchar] == '\0')
				nchar++;
			scales_name = (&argv[1][nchar]);
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;
		case '?':
			while (argv[1][nchar] == '\0')
				nchar++;
			Usage(program_name);
			break;
		default:
			(void)fprintf(stderr, "Bad option %s\n", argv[1]);
			Usage(program_name);
			error = false;
			break;

		}
		// move the argument list up one and move the count down one
		argv++; argc--;
	}

	if (error)
	{
		cout << "Bad option %s\n" << argv[1] << endl;;
		Usage(program_name);
	}

	if (cs_name == NULL)
	{
		cs_name = (char*)malloc(sizeof(cs_array));
		cs_name = (char*)cs_array;

	}

	if (scales_name == NULL)
	{
		scales_name = (char*)malloc(sizeof(scales_array));
		scales_name = (char*)scales_array;
	}
	
	if (norm_name == NULL)
	{
		norm_name = (char*)malloc(sizeof(minmax_array));
		norm_name = (char*)minmax_array;

	}

	if (labels_name == NULL)
	{
		labels_name = (char*)malloc(sizeof(lbls_array));
		labels_name = (char*)lbls_array;

	}

	if (pattern == NULL)
	{
		pattern = (char*)malloc(sizeof(pattern_array));
		pattern = (char*)pattern_array;

	}


	// write args on data structure
	p->cs_name = cs_name;
	p->quant_name = quant_name;
	p->labels_name = labels_name;
	p->norm_name = norm_name;
	p->scales_name = scales_name;
	p->number_of_candidates = number_of_candidates; 
	p->pattern = pattern;
	p->verbose = (bool) verbose;
	p->normalized_codebook = normalized_codebook;
	
}

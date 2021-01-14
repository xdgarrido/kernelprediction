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
	cout << "Usage is %s [options] :" << program_name << endl;
	cout << "Options" << endl;
	cout << "  -i input file name" << endl;
	cout << "  -j trainning set file name" << endl;
	cout << "  -g quantizer file name" << endl;
	cout << "  -k verification set file name" << endl;
	cout << "  -v verbose" << endl;
	cout << "  -n number of clusters " << endl;
	cout << "  -f 0: naive pnn 1: fast pnn " << endl;
	cout << "  -m 0: do not apply normalization 1: apply min-max normalization" << endl;
	cout << "  -p number of clusters used for the cs set" << endl;
	cout << "  -s start index representing dimensions to be removed from input data" << endl;
	cout << "  -e end index representing dimensions to be removed from input data" << endl;
	cout << "  -d index where the label field start" << endl;

	exit(1);
}


void ParseArgs(int argc, char *argv[], Args_t *p)
{
	char *program_name, *in_name=NULL, *ts_name=NULL, *cs_name=NULL, *cs_norm_name=NULL, *lbls_name=NULL, *quant_name=NULL, *no_of_clusters=NULL;
	int verbose=0, number_of_clusters=1024, clustering_type=1;
	std::tuple<int, int> removed_dimensions(0,0);
	int start_idx = 5;
	int end_idx = 14; 
	int label_idx = 5;
	int test_set_size = 1000; 
	bool error = true;
	int normalize_data= 1;
	char nchar = 2;
	const char* cs_array = "cs.csv";
	const char* cs_norm_array = "cs_norm.csv";
	const char* ts_array = "ts.csv";
	const char* lbls_array = "labels.csv";
	const char* quant_array = "quant";

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
			in_name = &argv[1][nchar];
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;
		case 'j':
			while (argv[1][nchar] == '\0')
				nchar++;
			ts_name = &argv[1][nchar];
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;
		case 'k':
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
		case 'v':
			while (argv[1][nchar] == '\0')
				nchar++;
			verbose = atoi(&argv[1][nchar]);
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;
		case 'p':
			while (argv[1][nchar] == '\0')
				nchar++;
			test_set_size = atoi(&argv[1][nchar]);
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;

		case 'f':
			while (argv[1][nchar] == '\0')
				nchar++;
			clustering_type = atoi(&argv[1][nchar]);
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;
		case 'n':
			while (argv[1][nchar] == '\0')
				nchar++;
			no_of_clusters = &argv[1][nchar];
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;

		case 'm':
			while (argv[1][nchar] == '\0')
				nchar++;
			normalize_data = atoi(&argv[1][nchar]);
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;

		case 's':
			while (argv[1][nchar] == '\0')
				nchar++;
			start_idx = atoi(&argv[1][nchar]);
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;

		case 'e':
			while (argv[1][nchar] == '\0')
				nchar++;
			end_idx = atoi(&argv[1][nchar]);
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;

		case 'd':
			while (argv[1][nchar] == '\0')
				nchar++;
			label_idx = atoi(&argv[1][nchar]);
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;

		case '?':
			while (argv[1][nchar] == '\0')
				nchar++;
			Usage(program_name);
			break;
		default:
			cout <<  "Bad option %s\n" << argv[1] << endl;
			Usage(program_name);
			error = false;
			break;

		}
		// move the argument list up one and move the count down one
		argv++; argc--;
	}

	
	if (error)
	{
		cout <<  "Bad option %s\n" << argv[1] << endl;
		Usage(program_name);
	}
	// Border bound checking parameters

	if (in_name == NULL)
	{
		cout << "Bad option %s\n" << argv[1] << endl;
		Usage(program_name);
	}

	if (cs_name == NULL)
	{
		cs_name = (char*)malloc(sizeof(cs_array));
		cs_name = (char*)cs_array;

	}

	if(cs_norm_name == NULL)
	{
		cs_norm_name = (char*)malloc(sizeof(cs_norm_array));
		cs_norm_name = (char*)cs_norm_array;

	}

	if (ts_name == NULL)
	{
		ts_name = (char*)malloc(sizeof(ts_array));
		ts_name = (char*)ts_array;

	}

	if (lbls_name == NULL)
	{
		lbls_name = (char*)malloc(sizeof(lbls_array));
		lbls_name = (char*)lbls_array;

	}

	if (quant_name == NULL)
	{
		quant_name = (char*)malloc(sizeof(quant_array));
		quant_name = (char*)quant_array;

	}

	std::get<0>(removed_dimensions) = start_idx;
	std::get<1>(removed_dimensions) = end_idx;

	// write args on data structure
	p->in_name = in_name;
	p->ts_name = ts_name;
	p->cs_name = cs_name;
	p->cs_norm_name = cs_norm_name;
	p->lbls_name = lbls_name;
	p->quant_name = quant_name;
	p->removed_dimensions = removed_dimensions;
	p->label_idx = label_idx;
	p->test_set_size = test_set_size;
	p->verbose = (bool) verbose;
	p->normalize_data = (bool) normalize_data;
	p->number_of_clusters = no_of_clusters;
	p->clustering_type = clustering_type;	
}

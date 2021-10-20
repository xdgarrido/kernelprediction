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
	cout << program_name << endl;
	cout << "Options" << endl;
	cout << "  -i input file name" << endl;
	cout << "  -o output file name" << endl;
	cout << "  -c convolution type" << endl;
	cout << "  -p precision (fp32 or fp16)" << endl;
	cout << "  -s convolution parameter size" << endl;
	cout << "  -f format (nhwc or nchw)" << endl;
	
	exit(1);
}


void ParseArgs(int argc, char *argv[], Args_t *p)
{
	char *program_name, *in_name=NULL, *out_name=NULL, *precision=NULL, *conv_type=NULL, *fmt=NULL;
	bool error = true;
	int convpar_size= 15; // convolution parameters
	char nchar = 2;
	const char* in_array = "conv.txt";
	const char* out_array = "test.sh";
	const char* convtype_array = "fwd";
	const char* precision_array = "fp32";
	const char* fmt_array = "NHWC"; 



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
		case 'o':
			while (argv[1][nchar] == '\0')
				nchar++;
			out_name = &argv[1][nchar];
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;
		case 'c':
			while (argv[1][nchar] == '\0')
				nchar++;
			conv_type = &argv[1][nchar];
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;
		case 'p':
			while (argv[1][nchar] == '\0')
				nchar++;
			precision = &argv[1][nchar];
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;
		
		case 'f':
			while (argv[1][nchar] == '\0')
				nchar++;
			fmt = &argv[1][nchar];
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
		Usage(program_name);
	}
	// Border bound checking parameters

	if (in_name == NULL)
	{
		cout << "Bad option %s\n" << argv[1] << endl;
		Usage(program_name);
	}

	if (in_name == NULL)
	{
		in_name = (char*)malloc(sizeof(in_array));
		in_name = (char*)in_array;

	}

	if (out_name == NULL)
	{
		out_name = (char*)malloc(sizeof(out_array));
		out_name = (char*)out_array;

	}

	if (precision == NULL)
	{
		precision = (char*)malloc(sizeof(precision_array));
		precision = (char*)precision_array;

	}

	if (conv_type == NULL)
	{
		conv_type = (char*)malloc(sizeof(convtype_array));
		conv_type = (char*)convtype_array;

	}

	if (fmt == NULL)
	{
		fmt = (char*)malloc(sizeof(fmt_array));
		fmt = (char*)fmt_array;

	}

	

	// write args on data structure
	p->in_name = in_name;
	p->out_name = out_name;
	p->precision = precision;
	p->conv_type = conv_type;
	p->convpar_size = convpar_size;
	p->format = fmt;
}


#include "parse.h"
using namespace std;
void Usage(char *program_name)
{
	cout << "Usage is %s [options]" << program_name << endl;
	cout << "Options:" << endl;
	cout << "  -i convolution parameters file name (ex: cs.csv)" << endl;
	cout << "  -g quantizer file name (ex: quant6000.txt)" << endl;
	cout << "  -t training set file name (ex: ts.csv)" << endl;
	cout << "  -m min_max scales file name (ex: domain.csv)" << endl;
	cout << "  -n number of predictors" << endl;
	cout << "  -v verbose" << endl;
	cout << "  -c checkpoint value to report progress (ex: 10000 ) "  << endl;
	cout << "  -l initial learning rate (ex: 0.00005 ) " << endl;
	cout << "  -d decay for the learning rate " << endl;
	cout << "  -r drop_rate for the learning rate (ex: 2000)" << endl;
	cout << "  -e number of epochs " << endl;
	cout << "  -w window, which defnes the midplane between 2 codevectors (used by lvq only) " << endl;
	exit(1);
}


void ParseArgs(int argc, char *argv[], Args_t *p)
{
	char *program_name = NULL;
	char *quant_name=NULL;
	char *cs_name = NULL;
	char *ts_name = NULL;
	char *minmax_name = NULL;
	// default names
	const char* cs_array      = "cs.csv";
	const char* minmax_array  = "domain.csv";
	const char* ts_array      = "ts.csv";
	int verbose = 0; 
	bool error = true;
	int number_of_candidates = 1;
	float window_distance = 0.3f;
	float learning_rate = 0.00005f;
	float decay =0.5f;
	int drop_rate = 2000;
	int epochs = 3;
	int checkpoint = 10000;
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
		case 't':
			while (argv[1][nchar] == '\0')
				nchar++;
			ts_name = &argv[1][nchar];
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;
		case 'm':
			while (argv[1][nchar] == '\0')
				nchar++;
			minmax_name = &argv[1][nchar];
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
		
		case 'd':
			while (argv[1][nchar] == '\0')
				nchar++;
			decay = (float) atof(&argv[1][nchar]);
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;

		case 'e':
			while (argv[1][nchar] == '\0')
				nchar++;
			epochs = atoi(&argv[1][nchar]);
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;

		case 'w':
			while (argv[1][nchar] == '\0')
				nchar++;
			window_distance = (float) atof(&argv[1][nchar]);
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;

		case 'c':
			while (argv[1][nchar] == '\0')
				nchar++;
			checkpoint = atoi(&argv[1][nchar]);
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;

		case 'r':
			while (argv[1][nchar] == '\0')
				nchar++;
			drop_rate = atoi(&argv[1][nchar]);
			if (nchar > 2) { argv++; argc--; }
			error = false;
			break;

		case 'l':
			while (argv[1][nchar] == '\0')
				nchar++;
			learning_rate = (float) atof(&argv[1][nchar]);
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
	
	if (minmax_name == NULL)
	{
		minmax_name = (char*)malloc(sizeof(minmax_array));
		minmax_name = (char*)minmax_array;

	}

	if (ts_name == NULL)
	{
		ts_name = (char*)malloc(sizeof(ts_array));
		ts_name = (char*)ts_array;

	}

	// write args on data structure
	p->cs_name = cs_name;
	p->quant_name = quant_name;
	p->ts_name = ts_name;
	p->minmax_name = minmax_name;
	p->number_of_candidates = number_of_candidates; 
	p->verbose = (bool) verbose;
	p->window_distance = window_distance;
	p->learning_rate = learning_rate;
	p->decay = decay;
	p->drop_rate = drop_rate;
	p->epochs = epochs;
	p->checkpoint = checkpoint;	
}

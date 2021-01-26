#include "learning.h"
#define PRECISION_DIGITS 10

vector<float> normalize_codes(vector<float> codes, vector<vector<float>> min_max, bool normalize)
{
    float cmin;
    float cmax;
    float delta;
    float inv_delta;
    int codes_size = (int) codes.size();
    vector<float> ncodes(codes_size);

    for (int i = 0; i < codes_size-1; i++)
    {
        if (normalize)
        {
            cmin = min_max[i][0];
            cmax = min_max[i][1];
            delta = cmax - cmin;

            if (delta != 0.)
            {
                inv_delta = (float)(1. / delta);
            }
            else
            {
                inv_delta = 1.;
            }
            ncodes[i] = inv_delta * ((float)codes[i] - cmin);
        }
        else
        {
            ncodes[i] = (float)codes[i];
        }
    }
    ncodes[codes_size - 1] = (float) codes[codes_size - 1];
    return(ncodes);
}


vector<vector<float>> fread_codes(string codes_set)
{
    vector<vector<float>> codes;
    ifstream scope(codes_set); 
    string line;
    

    if (scope.is_open() == false)
    {
        cout << "Unable to open file:" << codes_set;
        exit(-1);
    }
    vector<string> line_buffer;

    string word;

    while(!scope.eof())
    {
        string substr;
  
        scope >> line;

        vector<string> result;
        stringstream s_stream(line); //create string stream from the string
        int count = 0;
        while (s_stream.good()) {
            
            string substr;
            getline(s_stream, substr, ','); //get first string delimited by comma
            count++;
            result.push_back(substr);
        }
        vector<float> vect;

        for (int i = 0; i < count; i++)
        {
            word = result.at(i);
            vect.push_back((float)atof(word.c_str()));
        }
        // store in ts codebook
        codes.push_back(vect);
    }
    return codes;
}

void fprint_codes(char* codes_file, vector<vector<float>> codes)
{
   
    int codebook_size = (int)codes.size();

    if (codebook_size != 0)
    {
        ofstream out(codes_file, ofstream::out);
        out.precision(PRECISION_DIGITS);
        int no_of_dimensions = (int)codes[0].size();
        for (int i = 0; i < codebook_size - 1; i++)
        {
            for (int j = 0; j < no_of_dimensions - 1; j++)
            {
                float val = codes[i][j];
                out << val << ",";
            }
            float val = codes[i][no_of_dimensions - 1];
            out << val << endl;
        }
        for (int j = 0; j < no_of_dimensions - 1; j++)
        {
            float val = codes[codebook_size - 1][j];
            out << val << ",";
        }
        float val = codes[codebook_size - 1][no_of_dimensions - 1];
        out << val;
        out.close();
    }

}
void removeSubstrs(string& s, string& p) {
    string::size_type n = p.length();
    for (string::size_type i = s.find(p);
        i != string::npos;
        i = s.find(p))
        s.erase(i, n);
}

int main(int argc, char** argv)
{
    FILE* fd_in = NULL;
    Args_t repository, * pArgs;
    LearningArgs_t learning_params;
    char *fname_quant, *fname_cs, *fname_ts, *fname_minmax;

    
    
    // parse program input arguments
    pArgs = &repository;
    ParseArgs(argc, argv, pArgs);

 
    fname_quant = pArgs->quant_name;
    fname_minmax = pArgs->minmax_name;
    fname_ts = pArgs->ts_name;
    fname_cs    = pArgs->cs_name;
    int number_of_candidates = pArgs->number_of_candidates; 
    
    learning_params.learning_rate_start  = pArgs->learning_rate_start;
    learning_params.learning_rate_end  = pArgs->learning_rate_end;
    learning_params.decay = pArgs->decay;
    learning_params.drop_rate = pArgs->drop_rate;
    learning_params.epochs = pArgs->epochs;
    learning_params.checkpoint = pArgs->checkpoint;
    string learning_function(pArgs->learning_function);
    learning_params.learning_function = learning_function;

    string quant_set(fname_quant);
    string ts_set(fname_ts);
    string cs_set(fname_cs);
    string minmax_set(fname_minmax);

    vector<vector<float>> min_max    = fread_codes(minmax_set);
    vector<vector<float>> qs_codes   = fread_codes(quant_set);
    vector<vector<float>> cs_codes   = fread_codes(cs_set);
    vector<vector<float>> ts_codes   = fread_codes(ts_set);
    vector<vector<float>> csn_codes;

    // build filename output

    string tmp(fname_quant);
    string pattern = ".csv";
    removeSubstrs(tmp,pattern);
    tmp += "_opt.csv";
    const char* fname_optquant = tmp.c_str();


 
    // normalize test set
    for (int i = 0; i < (int) cs_codes.size(); i++)
    {
        vector<float> codes = cs_codes[i];
        vector<float> ncodes = normalize_codes(codes, min_max, true);
        csn_codes.push_back(ncodes);
    }

    if (glvq(qs_codes, ts_codes, csn_codes, learning_params, number_of_candidates) == -1)
      return(-1);
    
    fprint_codes((char*)fname_optquant, qs_codes);
  
    return 0;
}
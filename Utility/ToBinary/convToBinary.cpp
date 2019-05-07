// Author: Hengquan Zhang
// This file contains source code for converting coincidence data in txt format to binary format.

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <list>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <limits>
#include <sys/time.h>

using namespace std;

// split a string s to multiple strings (delimiter: c).
vector<string> explode(string s, char c)
{
	string buff="";
	vector<string> v;
	char n;
	
	for(unsigned i=0; i<s.length(); ++i)
	{
		n=s.at(i);
		if(n != c) buff+=n; else
		if(n == c && buff != "") { v.push_back(buff); buff = ""; }
	}
	if(buff != "") v.push_back(buff);
	
	return v;
}

int main(int argc, char * argv[]){

	ifstream fin;
	string filein = argv[1];
	fin.open(filein.c_str());
	ofstream fout;
	string fileout = filein.append(".binary");
	fout.open(fileout.c_str(), ios::out | ios::binary);
	int numline = 0;
	string line;

	// fin is the input txt file containing coincidence data. Each line has 6 numbers representing the coordinates of two photon interactions.
	// fout is the output binary file containing coincidence data. 
	if (fin.is_open() && fout.is_open()){

		while ( getline (fin, line) )
		{
			vector<string> l= explode(line,' ');
			numline += 1;
			if(numline < 6) cout<<line<<endl;
			if(l.size() != 6) {cout<<line<<" length is not 6!!"<<endl; continue;}	//make sure the line contains complete lor.
			float x[6];
			x[0] = strtof((l[0]).c_str(),0) ;
			x[1] = strtof((l[1]).c_str(),0) ;
			x[2] = strtof((l[2]).c_str(),0) ;
			x[3] = strtof((l[3]).c_str(),0) ;
			x[4] = strtof((l[4]).c_str(),0) ;
			x[5] = strtof((l[5]).c_str(),0) ;
			fout.write( (char*)x, 6 * sizeof(float));
	
		}
	}
	else cout<<"Unable to open input lor file!!"<<endl;

	cout<<numline<<" LOR"<<endl;
	fin.close();
	fout.close();

	// output the first 5 lines for checking the correctness.
	ifstream test;
	test.open(fileout.c_str(), ios::in | ios::binary);
	numline = 0;

	while(!test.eof())
	{
		numline += 1;
		if(numline >= 6) exit(0);
		float x[6];
		test.read((char*)x, 6 * sizeof(float));
		for(int i=0; i<6; i++) cout<<x[i]<<" ";
		endl(cout);
	}

	test.close();
	return 0;
}

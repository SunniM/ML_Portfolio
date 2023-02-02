#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
using std::cout;
using std::endl;

void print_stats(std::vector<double> &v);
double covar(std::vector<double> &rm, std::vector<double> &medv);
double cor(std::vector<double> &rm, std::vector<double> &medv);

int main(int argc, char **argv)
{
	std::ifstream inFS;
	std::string line;
	std::string rm_in, medv_in;
	const int MAX_LEN = 1000;
	std::vector<double> rm(MAX_LEN);
	std::vector<double> medv(MAX_LEN);

	cout << "Opening file Boston.csv: \n";

	inFS.open("Boston.csv");
	if (!inFS.is_open())
	{
		cout << "Could not open file Boston.csv\n";
		return 1;
	}

	cout << "Reading line 1\n";
	getline(inFS, line);

	cout << "heading: " << line << endl;

	int numObservations = 0;
	while (inFS.good())
	{
		getline(inFS, rm_in, ',');
		getline(inFS, medv_in, '\n');

		rm.at(numObservations) = stof(rm_in);
		medv.at(numObservations) = stof(medv_in);
		numObservations++;
	}
	rm.resize(numObservations);
	medv.resize(numObservations);

	cout << "New Length: " << rm.size() << endl;

	cout << "Closing file Boston.csv.\n";
	inFS.close();

	cout << "Number of records: " << numObservations << endl;

	cout << "\nStats for rm" << endl;
	print_stats(rm);

	cout << "\nStats for medv" << endl;
	print_stats(medv);

	cout << "\nCovariance = " << covar(rm, medv) << endl;
	cout << "\nCorrelation = " << cor(rm, medv) << endl;

	cout << "Program terminated.";

	return 0;
}
double sum(std::vector<double> &v)
{
	double total = 0;
	for (int i = 0; i < v.size(); i++)
		total += v[i];
	return total;
}
double mean(std::vector<double> &v)
{
	return sum(v) / v.size();
}

double median(std::vector<double> v)
{
	int size = v.size();
	std::sort(v.begin(), v.end());
	if (size == 0)
		return 0;
	else if (size % 2 == 0)
	{
		return (double)(v[size / 2 - 1] + v[size / 2]) / 2;
	}
	else
		return v[size / 2];
}
double range(std::vector<double> v)
{
	std::sort(v.begin(), v.end());
	return v[v.size() - 1] - v[0];
}

double covar(std::vector<double> &rm, std::vector<double> &medv)
{
	int n = rm.size();

	double rm_bar = mean(rm);
	double medv_bar = mean(medv);
	double numerator = 0;
	for (int i = 0; i < n; i++)
	{
		numerator += (rm[i] - rm_bar) * (medv[i] - medv_bar);
	}
	return numerator / (n - 1);
}
double cor(std::vector<double> &rm, std::vector<double> &medv)
{
	int n = rm.size();

	double rm_bar = mean(rm);
	double medv_bar = mean(medv);
	double numerator = 0;
	for (int i = 0; i < n; i++)
	{
		numerator += (rm[i] - rm_bar) * (medv[i] - medv_bar);
	}
	double rm_sqsum = 0;
	double medv_sqsum = 0;
	for (int i = 0; i < n; i++)
	{
		rm_sqsum += std::pow((rm[i] - rm_bar), 2);
	}
	for (int i = 0; i < n; i++)
	{
		medv_sqsum += std::pow((medv[i] - medv_bar), 2);
	}
	return numerator / std::sqrt(rm_sqsum * medv_sqsum);
}

void print_stats(std::vector<double> &v)
{
	cout << "Sum: " << sum(v) << endl;
	cout << "Mean: " << mean(v) << endl;
	cout << "Median: " << median(v) << endl;
	cout << "Range: " << range(v) << endl;
}

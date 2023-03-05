#define _USE_MATH_DEFINES

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace std;

template <typename T>
void train_test_split(vector<T> src, vector<T> &train, vector<T> &test, int train_amount);

template <typename T>
double accuracy(vector<T> y_pred, vector<T> y_act);
double specificity(vector<int> y_pred, vector<int> y_act);
double sensitivity(vector<int> y_pred, vector<int> y_act);
// vector<double> age, vector<int> pclass,

class NaiveBayes {
   public:
    double apriori[2] = {0};
    double sex_likelihood[2][2] = {0};
    double pclass_likelihood[3][2] = {0};
    double mean[2] = {0};
    double variance[2] = {0};
    vector<vector<double>> raw_prob;

    NaiveBayes() {
    }
    void fit(vector<int> sex, vector<double> age, vector<int> pclass, vector<int> survived) {
        vector<double> count(2, 0);
        for (int i = 0; i < survived.size(); i++) {
            if (survived[i] == 0) {
                count[0]++;
            } else {
                count[1]++;
            }
        }
        apriori[0] = count[0] / survived.size();
        apriori[1] = count[1] / survived.size();

        for (int i = 0; i < 2; i++) {  // sex liklihood
            for (int j = 0; j < 2; j++) {
                int c = 0;
                for (int k = 0; k < sex.size(); k++) {
                    if (sex[k] == i && survived[k] == j) {
                        sex_likelihood[i][j]++;
                    }
                }
                sex_likelihood[i][j] = sex_likelihood[i][j] / count[j];
            }
        }

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < pclass.size(); k++) {
                    if (pclass[k] == i + 1 && survived[k] == j) {
                        pclass_likelihood[i][j]++;
                    }
                }
                pclass_likelihood[i][j] = pclass_likelihood[i][j] / count[j];
            }
        }

        double sum[2] = {0};

        for (int i = 0; i < age.size(); i++) {
            sum[(survived[i] == 0 ? 0 : 1)] += age[i];
        }
        mean[0] = sum[0] / count[0];
        mean[1] = sum[1] / count[1];

        for (int i = 0; i < age.size(); i++) {
            int j = (survived[i] == 0 ? 0 : 1);
            sum[j] += pow(age[i] - mean[j], 2);
        }

        variance[0] = sum[0] / count[0];
        variance[1] = sum[1] / count[1];
    }
    vector<int> predict(vector<int> sex, vector<double> age, vector<int> pclass, vector<int> survived) {
        vector<int> y_pred;

        for (int i = 0; i < sex.size(); i++) {
            raw_prob.push_back(calculate_raw_prob(pclass[i], sex[i], age[i]));
            y_pred.push_back((raw_prob[i][0] > raw_prob[i][1] ? 0 : 1));
        }
        return y_pred;
    }
    void printValues() {
        cout << "Sex Likelihood: \n";
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                cout << sex_likelihood[i][j] << "\t";
            }
            cout << endl;
        }
        cout << endl;
        cout << "Class Likelihood: \n";
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                cout << pclass_likelihood[i][j] << "\t";
            }
            cout << endl;
        }
        cout << endl;
        cout << "Age Mean: " << endl;
        cout << mean[0] << "  " << mean[1] << endl
             << endl;

        cout << "Age Variance: " << endl;
        cout << variance[0] << "  " << variance[1] << endl
             << endl;
    }

   private:
    vector<double> calculate_raw_prob(int pclass, int sex, double age) {
        vector<double> probs;
        double num_s = pclass_likelihood[pclass - 1][1] * sex_likelihood[sex - 1][1] * apriori[1] * prob_density(age, 1);
        double num_p = pclass_likelihood[pclass - 1][0] * sex_likelihood[sex - 1][0] * apriori[0] * prob_density(age, 0);
        double denom = num_s + num_p;
        probs.push_back(num_s / denom);
        probs.push_back(num_p / denom);
        return probs;
    }
    double prob_density(double age, int i) {
        double t1 = 1 / sqrt(2 * M_PI * variance[i]);
        double t2 = exp(-(pow(age - mean[i], 2) / (2 * variance[i])));
        return t1 * t2;
    }
};

int main(int argc, char **argv) {
    string filename = "titanic_project.csv";
    ifstream inFS;
    string line;
    string id_in, p_class_in, survived_in, sex_in, age_in;
    const int MAX_LEN = 1050;

    vector<string> id(MAX_LEN);
    vector<int> pclass(MAX_LEN);
    vector<int> survived(MAX_LEN);
    vector<int> sex(MAX_LEN);
    vector<double> age(MAX_LEN);

    cout << "Opening file " << filename << ": \n";

    inFS.open(filename);
    if (!inFS.is_open()) {
        cout << "Could not open " << filename << "\n";
        return 1;
    }

    cout << "Reading line 1\n";
    getline(inFS, line);

    cout << "heading: " << line << endl;

    int numObservations = 0;
    while (inFS.good()) {
        getline(inFS, id_in, ',');
        id_in.erase(remove(id_in.begin(), id_in.end(), '\"'), id_in.end());
        getline(inFS, p_class_in, ',');
        getline(inFS, survived_in, ',');
        getline(inFS, sex_in, ',');
        getline(inFS, age_in, '\n');

        id.at(numObservations) = id_in;
        pclass.at(numObservations) = stoi(p_class_in);
        survived.at(numObservations) = stoi(survived_in);
        sex.at(numObservations) = stoi(sex_in);
        age.at(numObservations) = stof(age_in);

        numObservations++;
    }
    id.resize(numObservations);
    pclass.resize(numObservations);
    survived.resize(numObservations);
    sex.resize(numObservations);
    age.resize(numObservations);

    cout << "New Length: " << id.size() << endl;

    cout << "Closing file titanic_project.csv.\n\n";
    inFS.close();

    vector<double> age_train, age_test;
    vector<int> sex_train, sex_test, survived_train, survived_test, pclass_train, pclass_test;

    train_test_split(sex, sex_train, sex_test, 800);
    train_test_split(age, age_train, age_test, 800);
    train_test_split(pclass, pclass_train, pclass_test, 800);
    train_test_split(survived, survived_train, survived_test, 800);

    NaiveBayes model = NaiveBayes();
    std::chrono::time_point<std::chrono::system_clock> start, end;

    start = chrono::system_clock::now();
    model.fit(sex_train, age_train, pclass_train, survived_train);
    end = chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "training time: " << elapsed_seconds.count() << "s\n\n";
    model.printValues();
    vector<int> y_pred = model.predict(sex_test, age_test, pclass_test, survived_test);

    cout << "accuracy: " << accuracy(y_pred, survived_test) << endl;
    cout << "sensitivity: " << sensitivity(y_pred, survived_test) << endl;
    cout << "specificity: " << specificity(y_pred, survived_test) << endl;
}

template <typename T>
void train_test_split(vector<T> src, vector<T> &train, vector<T> &test, int train_amount) {
    for (int i = 0; i < train_amount; i++) {
        train.push_back(src.at(i));
    }
    for (int i = train_amount; i < src.size(); i++) {
        test.push_back(src.at(i));
    }
}
template <typename T>
double accuracy(vector<T> y_pred, vector<T> y_act) {
    int count = 0;
    int size = y_act.size();
    for (int i = 0; i < size; i++)
        if (y_pred[i] == y_act[i])
            count++;
    return (double)count / size;
}
double sensitivity(vector<int> y_pred, vector<int> y_act) {
    int size = y_act.size();

    int TP = 0;
    int FN = 0;
    for (int i = 0; i < size; i++)
        if (y_act[i] == 1) {
            if (y_pred[i] == 1)
                TP++;
            else
                FN++;
        }

    return (double)TP / (TP + FN);
}
double specificity(vector<int> y_pred, vector<int> y_act) {
    int size = y_act.size();
    int TN = 0;
    int FP = 0;
    for (int i = 0; i < size; i++)
        if (y_act[i] == 0) {
            if (y_pred[i] == 0)
                TN++;
            else
                FP++;
        }
    return (double)TN / (TN + FP);
}

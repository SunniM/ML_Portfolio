#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
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

// template <typename T>
class LogisticRegression {
   public:
    int n;
    double lr;
    int it;
    double w0;
    double w1;
    vector<vector<int>> features;

    LogisticRegression(double learning_rate, int iterations) {
        w0 = 1;
        w1 = 1;
        lr = learning_rate;
        it = iterations;
    }
    void fit(vector<int> predictor, vector<int> label) {
        n = label.size();
        features.push_back(vector(n, 1));
        features.push_back(predictor);

        for (int i = 0; i < it; i++) {
            vector<double> Z(n);

            for (int j = 0; j < n; j++) {
                double z = w0 * features[0][j] + w1 * features[1][j];
                Z[j] = sigmoid(z);
                double error = label[j] - Z[j];
                w0 += lr * error * features[0][j];
                w1 += lr * error * features[1][j];
            }
        }
    }
    vector<int> predict(vector<int> predictor) {
        vector<int> y_pred(predictor.size());
        for (int i = 0; i < predictor.size(); i++) {
            double z = sigmoid(w0 + w1 * predictor[i]);
            y_pred[i] = (z > .5 ? 1 : 0);
        }
        return y_pred;
    }

   private:
    double sigmoid(double z) {
        return 1 / (1 + exp(-z));
    }
};

int main(int argc, char **argv) {
    string filename = "titanic_project.csv";
    ifstream inFS;
    string line;
    string id_in, p_class_in, survived_in, sex_in, age_in;
    const int MAX_LEN = 1050;

    vector<string> id(MAX_LEN);
    vector<int> p_class(MAX_LEN);
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
        p_class.at(numObservations) = stoi(p_class_in);
        survived.at(numObservations) = stoi(survived_in);
        sex.at(numObservations) = stoi(sex_in);
        age.at(numObservations) = stof(age_in);

        numObservations++;
    }
    id.resize(numObservations);
    p_class.resize(numObservations);
    survived.resize(numObservations);
    sex.resize(numObservations);
    age.resize(numObservations);

    cout << "New Length: " << id.size() << endl;

    cout << "Closing file titanic_project.csv.\n\n\n";
    inFS.close();

    vector<int> survived_train, survived_test;
    vector<int> sex_train, sex_test;
    vector<double> age_train, age_test;

    train_test_split(sex, sex_train, sex_test, 800);
    train_test_split(survived, survived_train, survived_test, 800);

    LogisticRegression model = LogisticRegression(.001, 50000);

    std::chrono::time_point<std::chrono::system_clock> start, end;

    start = chrono::system_clock::now();
    model.fit(sex_train, survived_train);
    end = chrono::system_clock::now();
    chrono::duration<double> elapsed_seconds = end - start;
    time_t end_time = chrono::system_clock::to_time_t(end);

    cout << "training time: " << elapsed_seconds.count() << "s\n\n";

    cout << "Coefficients: " << endl;
    cout << "w0 = " << model.w0 << endl;
    cout << "w1 = " << model.w1 << endl << endl;

    vector<int> y_pred = model.predict(sex_test);


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

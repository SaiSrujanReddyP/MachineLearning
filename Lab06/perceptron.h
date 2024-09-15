#pragma once
#include<vector>
#include<iostream>
#include<stdlib.h>
#include<time.h>
#include<fstream>

using namespace std;
#define E 2.71828182845
#define BETA 0

//#define DISPLAY_SUM_SQUARE_ERRORS

enum activationFunctions { step, bipolar_step, sigmoid, tan_h, ReLU, leaky_ReLU };

class perceptron {
	vector<double> weights;	// weights.
	activationFunctions af;	// activation function being used
	int Ninput;				// number of inputs

public:
	vector<double> getWeights() {
		return weights;
	}
	double getWeights(int index) {
		return weights[index];
	}
	double& operator[](int index) {
		return weights[index];
	}
	void setWeights(int index, double value) {
		weights[index] = value;
	}
	perceptron() {
		srand(15);
		weights = { 0,1,10 };
		af = sigmoid;
		Ninput = 2;
	}
	perceptron(activationFunctions activationFunct, initializer_list<float> initialWeights, int numberOfInputs) :af(activationFunct), Ninput(numberOfInputs) {
		weights.resize(numberOfInputs + 1);
		weights.shrink_to_fit();
		copy(initialWeights.begin(), initialWeights.end(), weights.begin());
	}
	perceptron(activationFunctions activationFunct,  int numberOfInputs) :af(activationFunct), Ninput(numberOfInputs) {
		srand(1726229676);								// randomly chosen value.
		weights.resize(numberOfInputs + 1);
		weights.shrink_to_fit();
		for (int i = 0; i < weights.size(); i++)
			weights[i] = (rand() % 100) / 100.01 - 0.5; // initializing all weights to 0
		cout << '\n';
	}
	double summation(double* inputs) {
		double sum = 0;
		for (int i = 0; i < Ninput; i++)
			sum += float(inputs[i]) * weights[i + 1];
		return sum;
	}
	float calcError(int prediction, int output) {
		return output - prediction;
	}
	double activationFunction(double sum) {
		double sgmd;
		switch (af) {
		case step:
			return (sum > 0 ? 1 : 0);
			break;
		case bipolar_step:
			return (sum > 0 ? 1 : (sum < 0 ? -1 : 1));
			break;
		case sigmoid:
			sgmd =  1 / (1 + pow(E, -sum));
			return sgmd;
			break;
		case tan_h:
			return (pow(E, sum) - pow(E, -sum)) / (pow(E, sum) + pow(E, -sum));
			break;
		case ReLU:
			return sum > 0 ? sum : 0;
			break;
		case leaky_ReLU:
			return std::max(sum, sum * BETA);
			break;
		}
		std::cout << "no activation function/ invalid activation function";
		return -1;
	}
	double predict(double* inputs) {
		return activationFunction(summation(inputs) + weights[0]);
	}
	void train(int nEpochs, vector<double*>& inputs, vector<int>& outputs, float learningRate) {
#ifdef DISPLAY_SUM_SQUARE_ERRORS
		ofstream filewrite;
		filewrite.open("errors.csv");
#endif 

		vector<int> predictions(outputs.size());
		for (int i = 0; i < nEpochs; i++) {
#ifdef DISPLAY_SUM_SQUARE_ERRORS
			double error = 0;
#endif
			for (int j = 0; j < inputs.size(); j++) {
				predictions[j] = predict(inputs[j]);


#ifdef DISPLAY_SUM_SQUARE_ERRORS
				error += (predictions[j] - outputs[j]) * (predictions[j] - outputs[j]);
#endif
				weights[0] += learningRate * calcError(predictions[j], outputs[j]);
				for (int w = 1; w < weights.size(); w++) {
					weights[w] += learningRate * calcError(predictions[j], outputs[j]) * inputs[j][w - 1];
				}

			}
#ifdef DISPLAY_SUM_SQUARE_ERRORS
			cout << "error at epoch " << i << " = " << error << "\n";
			filewrite << i << "," << error << "\n";
#endif
		}
		for (int i = 0; i < inputs.size(); i++) {
			std::cout << "\nexpected = " << outputs[i] << "; predicted => " << predictions[i];
		}
#ifdef DISPLAY_SUM_SQUARE_ERRORS
		filewrite.close();
#endif

	}
	int train(vector<double*>& inputs, vector<int>& outputs, float learningRate) {	// keep going until error < 0.002
#ifdef DISPLAY_SUM_SQUARE_ERRORS
		ofstream filewrite;
		// writing to csv file for analysis
		filewrite.open("errors.csv");
#endif 

		vector<int> predictions(outputs.size());
		double error;
		int iterations = 0;
		do {
			error = 0;
			for (int j = 0; j < inputs.size(); j++) {
				predictions[j] = predict(inputs[j]);

				error += (predictions[j] - outputs[j]) * (predictions[j] - outputs[j]);
				weights[0] += learningRate * calcError(predictions[j], outputs[j]);
				for (int w = 1; w < weights.size(); w++) {
					weights[w] += learningRate * calcError(predictions[j], outputs[j]) * inputs[j][w - 1];
				}

			}
#ifdef DISPLAY_SUM_SQUARE_ERRORS
			filewrite << error << "\n";
#endif
			iterations++;
		} while (error > 0.002 && iterations < 10000);
		std::cout << "iterations = " << iterations;

		for (int i = 0; i < inputs.size(); i++) {
			std::cout << "\nexpected = " << outputs[i] << "; predicted => " << predictions[i];
		}
#ifdef DISPLAY_SUM_SQUARE_ERRORS
		filewrite.close();
#endif
		return iterations;
	}
};


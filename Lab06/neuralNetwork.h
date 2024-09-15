#pragma once
#include"perceptron.h"
#include<array>
class neuralNetwork
{
	int hiddenSize;		/* size of hidden layers */
	int n_inputs;		/* number of inputs */
	perceptron *hiddenLayers, outputLayer;
	double* hiddenDel, *hiddenPrediction;	/* del values for hidden layer neurons */
public:
	neuralNetwork(int hiddenLayerSize, int inputSize):hiddenSize(hiddenLayerSize),n_inputs(inputSize){
		hiddenLayers = new perceptron[hiddenLayerSize];
		outputLayer = perceptron(sigmoid, hiddenLayerSize);
		for (int i = 0; i < hiddenSize; i++) {
			hiddenLayers[i] = perceptron(sigmoid, inputSize);
		}
		/* below variables are only used for training/back propogation but re-allocation of memory will take lot of time, hence making it a class variable*/
		hiddenPrediction = new double[hiddenSize];
		hiddenDel = new double[hiddenSize];
	}
	~neuralNetwork() {
		delete[] hiddenLayers;
		delete[] hiddenPrediction;
		delete[] hiddenDel;
	}
	double predict(double* inputs) {
		for (int i = 0; i < hiddenSize; i++)
			hiddenPrediction[i] = hiddenLayers[i].predict(inputs);		
		return outputLayer.predict(hiddenPrediction);
	}
	void backPropogate(double* input,double requiredOutput, float learningRate) {
		/* in the back propogation the predict function is not used directly, but the code for it is used in it. */
		double outputDel;

		double output = predict(input);
		
		// calculating del values for every neuron, in order to update the weights.
		outputDel = output * (1 - output) * (requiredOutput - output);
		outputLayer.setWeights(0, outputLayer[0] + learningRate * outputDel);

		for (int i = 0; i < hiddenSize; i++) {
			hiddenDel[i] = hiddenPrediction[i] * (1 - hiddenPrediction[i]) * (outputLayer[i] * outputDel);	// outputLayer[i] => ith weight of output layer
			// setting new weights on output layer.
			outputLayer.setWeights(i + 1, outputLayer[i + 1] + learningRate * outputDel * hiddenPrediction[i]);

			//setting new weights for hidden layers
			hiddenLayers[i].setWeights(0, hiddenLayers[i][0] + learningRate * hiddenDel[i]);
	//		cout << "\nhidden layer " << i << " weights : " << hiddenLayers[i][0] << " ";
			for (int j = 0; j < n_inputs; j++) {
		//		cout << hiddenLayers[i][j + 1] << " ";
				hiddenLayers[i].setWeights(j + 1, hiddenLayers[i][j + 1] + learningRate * hiddenDel[i] * input[j]);
			}
		}
	}
	void train(vector<double*> inputs, double* outputs, int nEpochs, float learningRate = 1) {
		for (int epoch = 0; epoch < nEpochs; epoch++) {
			for (int i = 0; i < inputs.size(); i++) {
				backPropogate(inputs[i], outputs[i], learningRate);
			}
			if (epoch % 500 == 0) {
				cout << "\nat epoch " << epoch << ", loss : ";
				double loss = 0;
				for (int i = 0; i < 4; i++) {
					double pred = predict(inputs[i]);
					loss += pow(pred - outputs[i], 2);

				}
				loss /= 4;
				cout << loss;
			}
		}
	}
};


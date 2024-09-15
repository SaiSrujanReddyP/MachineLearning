
//#include "basicNN.h"
#include"neuralNetwork.h"
#include"rapidcsv.h"

using namespace std;

#define Q9

int main(){
#ifdef Q2
	perceptron perceptron1 = perceptron(sigmoid,{10,0.2,-0.75},2);

	double inputs[][2] = {{0,0},{0,1},{1,0},{1,1}};

	vector<double*> in(std::size(inputs));
		
	for(int i = 0; i < std::size(inputs); i++)
		in[i] = inputs[i];

	vector<int> outputs = {0,0,0,1};
	perceptron1.train(200,in,outputs, 0.05);
#endif
#ifdef Q3

	double inputs[][2] = { {0,0},{0,1},{1,0},{1,1} };
	vector<double*> in(std::size(inputs));
	for (int i = 0; i < std::size(inputs); i++)
		in[i] = inputs[i];
	vector<int> outputs = { 0,0,0,1 };

	cout << "\nusing bipolar step -- :\n";
	perceptron perceptron1 = perceptron(bipolar_step, { 10,0.2,-0.75 }, 2);
	perceptron1.train(300, in, outputs, 0.05);
	cout << "\nusing ReLU -- :\n";

	perceptron perceptron2 = perceptron(ReLU, { 10,0.2,-0.75 }, 2);
	perceptron2.train(300, in, outputs, 0.05);

	cout << "\nusing sigmoid -- :\n";
	perceptron perceptron3 = perceptron(sigmoid, { 10,0.2,-0.75 }, 2);
	perceptron3.train(300, in, outputs, 0.05);
#endif
#ifdef Q4

	double inputs[][2] = { {0,0},{0,1},{1,0},{1,1} };
	vector<double*> in(std::size(inputs));
	for (int i = 0; i < std::size(inputs); i++)
		in[i] = inputs[i];

	vector<int> outputs = { 0,0,0,1 };
	perceptron perceptron1;
	ofstream filewrite;
	// writing to csv file for analysis
	filewrite.open("epochs.csv");
	for (float i = 0.1; i < 1.1; i += 0.1) {
		perceptron1 = perceptron(step, { 10,0.2,-0.75 }, 2);
		filewrite << perceptron1.train(in, outputs, i) << "," << i << "\n";
		cin.get();
	}
	filewrite.close();
#endif
#ifdef Q5
	double inputs[][2] = { {0,0},{0,1},{1,0},{1,1} };
	vector<double*> in(std::size(inputs));
	for (int i = 0; i < std::size(inputs); i++)
		in[i] = inputs[i];
	vector<int> outputs = { 0,1,1,0 };

	cout << "\nusing bipolar step -- :\n";
	perceptron perceptron1 = perceptron(bipolar_step, { 10,0.2,-0.75 }, 2);
	perceptron1.train(300, in, outputs, 0.05);
	cout << "\nusing ReLU -- :\n";

	perceptron perceptron2 = perceptron(ReLU, { 10,0.2,-0.75 }, 2);
	perceptron2.train(300, in, outputs, 0.05);

	cout << "\nusing sigmoid -- :\n";
	perceptron perceptron3 = perceptron(sigmoid, { 10,0.2,-0.75 }, 2);
	perceptron3.train(300, in, outputs, 0.05);
#endif
#ifdef Q6
	// get data from csv file.
	rapidcsv::Document doc("data.csv");
	std::vector<int> candies = doc.GetColumn<int>("Candies");
	std::vector<int> mangoes = doc.GetColumn<int>("Mangoes");
	std::vector<int> milk = doc.GetColumn<int>("Milk");
	std::vector<int> packets = doc.GetColumn<int>("Packets");
	std::vector<std::string> high_val = doc.GetColumn<std::string>("high_val");

	// making perceptron to classify data.
	perceptron p1(sigmoid, { 10,-0.3,0.5, 100,0.9 }, 4);
	std::vector<int> outputs(candies.size());	// the size of any random columns is taken to see the number of rows totally.
	for (int i = 0; i < outputs.size(); i++)
		outputs[i] = (high_val[i] == "Yes" ? 1 : 0);

	std::vector<double*> inputs(candies.size());

	double** data = new double*[candies.size()];
	for (int i = 0; i < candies.size(); i++)
		data[i] = new double[4];

	for (int i = 0; i < candies.size(); i++) {
		data[i][0] = candies[i];
		data[i][1] = mangoes[i];
		data[i][2] = milk[i];
		data[i][3] = packets[i];
		inputs[i] = data[i];
	}

	p1.train(inputs, outputs, 0.05);
	

	for (int i = 0; i < candies.size(); i++)
		delete[] data[i];
	delete[] data;
	
#endif
#ifdef Q7
	// pseudo inverse method for predicting results for same data is done in lab2.py
#endif
#ifdef Q8
	double inputs[][2] = { {0,0},{0,1},{1,0},{1,1} };
	vector<double*> in(std::size(inputs));

	for (int i = 0; i < std::size(inputs); i++)
		in[i] = inputs[i];

	double outputs[] = {0,0,0,1};
	neuralNetwork n1(100, 2);
	n1.train(in, outputs, 10000);
	std::cout << '\n';
	for (int i = 0; i < 4; i++)
		std::cout << n1.predict(inputs[i]) << "\n";

#endif
#ifdef Q9
	double inputs[][2] = { {0,0},{0,1},{1,0},{1,1} };
	vector<double*> in(std::size(inputs));

	for (int i = 0; i < std::size(inputs); i++)
		in[i] = inputs[i];

	double outputs[] = { 0,1,1,0 };
	neuralNetwork n1(3, 2);
	n1.train(in, outputs, 100000,0.9);
	std::cout << '\n';
	for (int i = 0; i < 4; i++)
		std::cout << n1.predict(inputs[i]) << "\n";
#endif

}

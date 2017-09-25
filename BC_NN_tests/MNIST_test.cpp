
#include "Tensor.h"
#include "FeedForward.h"
#include "NeuralNetwork.h"
#include "ConvolutionalLayer.h"
#include <fstream>

typedef std::vector<Vector<double>> data;

Tensor<double> expandOutput(int val, int total) {
	Tensor<double> out(total);
	out.fill(0);
	out(val) = 1;
	return out;
}

Tensor<double> normalize(Tensor<double> tens, double max, double min) {
	for (unsigned i = 0; i < tens.size(); ++i) {
		tens(i) = (tens(i) - min) / (max - min);
	}
	return tens;
}

void generateAndLoad(data& input_data, data& output_data, std::ifstream& read_data, unsigned MAXVALS) {
	unsigned numb = 0;
	unsigned vals = 0;
	while (read_data.good() && vals < MAXVALS) {
		Tensor<double> input;
		input.readCSV(read_data, 785);

		input_data.push_back(normalize(input({1}, {784}), 255, 0));
		output_data.push_back(expandOutput(input(0), 10));

		++vals;
	}
}


int main() {
	Tensor<double> t;

	//Create neural network object
	NeuralNetwork net;

	//Create layers
	ConvolutionalLayer f1(28, 28, 1, 3, 3, 3);
	ConvolutionalLayer f2(26, 26, 3, 3, 3, 3);

	FeedForward f3(1728, 10);
	//add them to the nn
	net.add(&f1);
	net.add(&f2);
	net.add(&f3);

	//initialize data storageb
	data inputs(0);
	data outputs(0);

	data testInputs(0);
	data testOutputs(0);

	//load data
	std::cout << "loading data..." << std::endl;
	std::ifstream is("///home/joseph///Downloads///train.csv");
	generateAndLoad(inputs, outputs, is, 40000);
	generateAndLoad(testInputs, testOutputs, is, 1000);

	std::cout << "testing..." << std::endl;
	double test_error = 0;
	test_error = net.test(testInputs, testOutputs);
	std::cout << " test error post training -- " << test_error << std::endl;

	//train neural network
	std::cout << "training..." << std::endl;
	net.train(inputs, outputs, 10);

	//Test the error again
	std::cout << "testing..." << std::endl;
	 test_error = 0;
	test_error = net.test(testInputs, testOutputs);
	std::cout << " test error post training -- " << test_error << std::endl;


	return 0;
}

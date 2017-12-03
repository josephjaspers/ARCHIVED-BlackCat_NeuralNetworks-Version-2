#include "Tensor.h"
#include "FeedForward.h"
#include "NeuralNetwork.h"
#include "ConvolutionalLayer.h"
#include "MaxPoolingLayer.h"
#include "AdjustedConvolutionalLayer.h"
#include <fstream>

typedef std::vector<vec> data;

tensor expandOutput(int val, int total) {
	//Convert a value to a 1-hot output vector
	tensor out(total);
	out.zeros();
	out[val] = 1;
	return out;
}

tensor normalize(tensor tens, double max, double min) {
	//generic feature scaling (range of [0,1]
	tens -= min;
	tens /= (max - min);

	return tens;
}

void generateAndLoad(data& input_data, data& output_data, std::ifstream& read_data, unsigned MAXVALS) {
	unsigned vals = 0;
	while (read_data.good() && vals < MAXVALS) {
		tensor input;
		input.readCSV(read_data, 785);
		auto v = normalize(input({ 1 }, { 784 }), 255, 0);
		input_data.push_back(v);
		output_data.push_back(expandOutput(input(0).get(), 10));
		++vals;
	}
	std::cout << " return -- finished creating data set " << std::endl;
}

int ConvMNIST() {

	//Create neural network object
	NeuralNetwork net;

	//Create layers
	ConvolutionalLayer c1(28, 28, 1, 3, 3, 32);				//(img_rows, img_cols, img_depth, filt_rows, filt_cols, numb_filters)
	MaxPoolingLayer    p1(26, 26, 32, 2);
	ConvolutionalLayer c2(13, 13, 32, 4, 4, 16);					//(img_rows, img_cols, img_depth, stride)
	FeedForward 	   f1(1600, 10);							//(inputs, outputs)

	net.add(&c1);
	net.add(&p1);
	net.add(&c2);
	net.add(&f1);


	net.setLearningRate(.0001);  //default = .01

	//initialize data storage
	data inputs(0);
	data outputs(0);

	data testInputs(0);
	data testOutputs(0);

	//load data
	std::cout << "loading data..." << std::endl << std::endl;
	std::ifstream in_stream("///home/joseph///Downloads///train.csv");

	//Load 40,000 training examples (taken from kaggle digit recognizer train.csv)
	generateAndLoad(inputs, outputs, in_stream, 40000);
	//Load 1000 training exampels to be used as a test set
	generateAndLoad(testInputs, testOutputs, in_stream, 1000);

	std::cout << "testing initial error..." << std::endl << std::endl;
	double test_error = 0;
	test_error = net.test(testInputs, testOutputs);
	std::cout << " test error post training -- " << test_error << std::endl << std::endl;

	//train neural network
	unsigned NUMB_ITERATIONS = 20;
	std::cout << "training... --- numb epochs = " << NUMB_ITERATIONS << std::endl;
	net.realTimeTrain(inputs, outputs, NUMB_ITERATIONS, 1000); //prints out the current error in real time

	//Test the error again
	std::cout << "testing..." << std::endl << std::endl;
	std::cout << "Test error ----" << std::endl << std::endl;
	test_error = 0;
	test_error = net.test(testInputs, testOutputs);

	std::cout << std::endl << "Train error ----" << std::endl;

	test_error = net.test(inputs, outputs);

	std::cout << " test error post training -- " << test_error << std::endl;

	return 0;
}

int percept_MNIST() {

	std::cout << " percept _msnist " << std::endl;
	//Create neural network object
	NeuralNetwork net;

	//Create layers
	FeedForward f1(784, 250);
	FeedForward f3(250, 10);

	net.add(&f1);
	net.add(&f3);

	net.setLearningRate(.01);  //default = .01

	//initialize data storage
	data inputs(0);
	data outputs(0);

	data testInputs(0);
	data testOutputs(0);

	//load data
	std::cout << "loading data..." << std::endl << std::endl;
	std::ifstream in_stream("///home/joseph///Downloads///train.csv");

	//Load 40,000 training examples (taken from kaggle digit recognizer train.csv)
	generateAndLoad(inputs, outputs, in_stream, 40000);
	//Load 1000 training exampels to be used as a test set
	generateAndLoad(testInputs, testOutputs, in_stream, 2000);
	std::cout << "data loaded " << std::endl;


	std::cout << "testing initial error..." << std::endl << std::endl;
	double test_error = 5;
	test_error = net.test(testInputs, testOutputs);
	std::cout << " test error post training -- " << test_error << std::endl << std::endl;

	//train neural network
	unsigned NUMB_ITERATIONS = 10;
	std::cout << "training... --- numb epochs = " << NUMB_ITERATIONS << std::endl;
	net.realTimeTrain(inputs, outputs, NUMB_ITERATIONS, 1000); //prints out the current error in real time

	//Test the error again
	std::cout << "testing..." << std::endl << std::endl;
	std::cout << "Test error ----" << std::endl << std::endl;
	test_error = 0;
	test_error = net.test(testInputs, testOutputs);

	std::cout << std::endl << "Train error ----" << std::endl;

	test_error = net.test(inputs, outputs);
	std::cout << " test error post training -- " << test_error << std::endl;
	return 0;
}

int main() {

	percept_MNIST();
	//ConvMNIST();
	//ADJConvMNIST();
}

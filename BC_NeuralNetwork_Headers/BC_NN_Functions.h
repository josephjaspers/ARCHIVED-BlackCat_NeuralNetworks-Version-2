/*
 * BC_NN_Functions.h
 *
 *  Created on: Oct 7, 2017
 *      Author: joseph
 */

#ifndef BC_NN_FUNCTIONS_H_
#define BC_NN_FUNCTIONS_H_

namespace nonLin {


	std::pair<Tensor<double, CPU>, Tensor<unsigned, CPU>> max_pooling(Tensor<double, CPU>& x, unsigned stride) {
		Tensor<double> max_cpy = x.getShape();
		max_cpy.zeros();
		std::pair<Tensor<double, CPU>, Tensor<unsigned, CPU>> data;
		Tensor<unsigned, CPU> index(2, floor(x.rows() / stride), floor(x.cols() / stride), x.rank(2));
		index.zeros();
		Tensor<double, CPU> max_value(floor(x.rows() / stride), floor(x.cols() / stride), x.rank(2));

		for (unsigned z = 0; z < x.rank(2); ++z) {
			for (unsigned r = 0; r < x.rows(); r += stride) {
				for (unsigned c = 0; c < x.cols(); c += stride) {

					std::pair<Scalar<double, CPU>, Tensor<unsigned, CPU>> tmp = x( { z, r, c }, { stride, stride }).max_index();
					max_value[z][r / stride][c / stride] = tmp.first;
					index[z][r / stride][c / stride] = tmp.second;

				}
			}
		}

		data.first = max_value;
		data.second = index;

		return data;
	}

	Tensor<double, CPU> max_filling(Tensor<double, CPU>& x, Tensor<unsigned> id, unsigned stride) {

		Tensor<double> maximums = { x.rows() * stride, x.cols() * stride, x.rank(2) };
		maximums = 0;
		for (unsigned z = 0; z < x.rank(2); ++z) {
			for (unsigned r = 0; r < x.rows(); r++) {
				for (unsigned c = 0; c < x.cols(); c++) {
					unsigned row = id[z][r][c](0).get();
					unsigned col = id[z][r][c](1).get();

					maximums[z][r * stride + row][c * stride + col] = x[z][r](c);

				}
			}
		}

		return maximums;
	}


	void constrain(tensor& t, double l_bound, double u_bound) {
		for (int i = 0; i < t.size(); ++i) {
			if (t.data()[i] < l_bound)
			t.data()[i] = l_bound;
			else if (t.data()[i] > u_bound) {
				t.data()[i] =  u_bound;
			}
		}
	}

	Tensor<double> abs(Tensor<double> value) {
		Tensor<double> absolute = value;

		for (unsigned i = 0; i < absolute.size(); ++i) {
			if (absolute.data()[i] < 0) {
				absolute.data()[i] *= -1;
			}
		}
		return absolute;
	}


	//weights some MxNxDxK img M1xN1xD error is M2xN2xK
	Tensor<double> adj_convolution_filter_error(const Tensor<double>& weights,const Tensor<double>& img, Tensor<double> error) {


		//weights.print();
		//std::cout << " filter erro " << std::endl;
		Tensor<double> krnl_error = weights.getShape();
		krnl_error.zeros();
		Tensor<double> ones = weights[0][0].getShape();
		ones.fill(1);
		constrain(error, -1, 1);


		unsigned error_depth = img.rank(2);
		unsigned filter_depth = weights.rank(3);

		for (unsigned f = 0; f < filter_depth; ++f) {
			for (unsigned z = 0; z < error_depth; ++z) {
				for (unsigned r = 0; r < error.rows(); ++r) {
					for (unsigned c = 0; c < error.cols(); ++c) {
						//error[z][r][c]

						///range of [0,1]
						//std::cout << " diff " << std::endl;
						Tensor<double> difference = img({z,r,c},{weights.rows(), weights.cols()}) - weights[f][z];
					//	img({z,r,c},{weights.rows(), weights.cols()}).print();
					//	weights[f][z].print();
					//	difference.print();
					//	error[z][r](c).print();

						//std::cout << " next " << std::endl;
						if (error[z][r].data()[c] < 0) {

							krnl_error[f][z] += ((ones - abs(difference)) & error[z][r](c));
						} else {
							krnl_error[f][z] += difference & error[z][r](c); //multiply the difference by the error
						}
					//	krnl_error[f][z].print();
					//	int wait;
					//	std::cin >> wait;
					}
				}
			}
		}
		return krnl_error;
	}
	Tensor<double> adj_convolution_img_error(Tensor<double>& weights, Tensor<double> img, Tensor<double> error) {

		Tensor<double> img_error = img.getShape();
		img_error.zeros();
		Tensor<double> ones = weights[0][0].getShape();
		ones.fill(1);
		constrain(error, -1, 1);

		unsigned error_depth = img.rank(2);
		unsigned filter_depth = weights.rank(3);

		for (unsigned f = 0; f < filter_depth; ++f) {
			for (unsigned z = 0; z < error_depth; ++z) {
				for (unsigned r = 0; r < error.rows(); ++r) {
					for (unsigned c = 0; c < error.cols(); ++c) {
						//error[z][r][c]

						///range of [0,1]
						//std::cout << " diff " << std::endl;
						Tensor<double> difference = img({z,r,c},{weights.rows(), weights.cols()}) - weights[f][z];

						//std::cout << " next " << std::endl;
						if (error[z][r].data()[c] < 0) {
							//std::cout << " hereo " << std::endl;
							img_error({z,r,c},{weights.rows(), weights.cols()}) += ((ones - abs(difference)) & error[z][r](c));
						} else {
							//std::cout << " 2 " << std::endl;
							img_error({z,r,c},{weights.rows(), weights.cols()}) += difference & error[z][r](c); //multiply the difference by the error
						}
					}
				}
			}
		}
		return img_error;
	}
	void sigmoid(Tensor<double, CPU>& x) {
		for (unsigned i = 0; i < x.size(); ++i) {
			x.data()[i] = 1 / (1 + pow(2.71828, -x.data()[i]));
		}
	}

	void sigmoid_deriv(Tensor<double, CPU>& x) {
		for (unsigned i = 0; i < x.size(); ++i) {
			x.data()[i] *= (1 - x.data()[i]);
		}
	}
};

#endif /* BC_NN_FUNCTIONS_H_ */

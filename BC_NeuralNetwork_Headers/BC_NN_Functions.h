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

	void sigmoid(Tensor<double, CPU>& x) {
		for (int i = 0; i < x.size(); ++i) {
			x.data()[i] = 1 / (1 + pow(2.71828, -x.data()[i]));
		}
	}

	void sigmoid_deriv(Tensor<double, CPU>& x) {
		for (int i = 0; i < x.size(); ++i) {
			x.data()[i] *= (1 - x.data()[i]);
		}
	}

	void sigmoid(Tensor<float, GPU>& x);
	void sigmoid_deriv(Tensor<float, GPU>& x);


};

#endif /* BC_NN_FUNCTIONS_H_ */

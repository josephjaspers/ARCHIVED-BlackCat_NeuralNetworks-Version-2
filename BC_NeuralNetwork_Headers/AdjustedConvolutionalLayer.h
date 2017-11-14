#ifndef BLACKCAT_ADJConvolutionalLayer_h
#define BLACKCAT_ADJConvolutionalLayer_h

#include "Layer.h"

class AdjustedConvolutionalLayer : public Layer {
public:
	unsigned img_r;			//numb rows of input signal
	unsigned img_c;			//numb cols of input signal
	unsigned img_d;			//depth of the input signal

	unsigned numb_krnls;	//number of filters or kernels
	unsigned krnl_r;		//number of rows of the kernels
	unsigned krnl_c;		//number of cols of the kernels
	unsigned krnl_d;		//depth of the kernels

	unsigned out_r;			//output rows
	unsigned out_c;			//output cols
	unsigned out_d;			//output depth

	unsigned r_pos;			//number of rows for correlation/convolution positions
	unsigned c_pos;			//number of cols for correlation/convolution positions

	tensor y;					//output
	tensor w;  					//filter
	tensor dx;
	tensor w_gradientStorage;

public:

	AdjustedConvolutionalLayer(unsigned img_rows, unsigned img_cols, unsigned depth, unsigned filt_rows, unsigned filt_cols, unsigned n_filters) {
		this->img_d = depth;
		this->img_r = img_rows;
		this->img_c = img_cols;

		this->krnl_d = depth;
		this->numb_krnls = n_filters;
		this->krnl_r = filt_rows;
		this->krnl_c = filt_cols;

		this->out_r = img_rows - filt_rows + 1;
		this->out_c = img_cols - filt_cols + 1;
		this->out_d = n_filters;

		r_pos = img_rows - krnl_r + 1;
		c_pos = img_cols - krnl_c + 1;

		w = tensor(krnl_r, krnl_c, img_d, numb_krnls);

		w_gradientStorage = tensor(krnl_r, krnl_c, img_d, numb_krnls);

		w.randomize(0, 1);

		y = tensor(out_r, out_c, numb_krnls);
		dx = tensor(img_r, img_c, img_d);
	}

	void printImg(tensor& x) {

		for (int z = 1; z < x.rank(2); ++z) {
			for (int r = 0; r < x.rank(1); ++r) {
				for (int c = 0; c< x.rank(0); ++c) {
					if (x[z][r](c).get() > .001) {
						auto str =std::to_string(x[z][r](c).get());
									str = str.substr(0, str.length() < 5 ? str.length() : 5);
									std::cout << str << " ";
					} else{
						std::cout << "      ";
					}
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;

		}
	}

	virtual vec forwardPropagation(vec data) {
		//std::cout << " fp " << std::endl;

		tensor x = data;
		x.reshape( { img_r, img_c, img_d });
		y = (w.ADJUSTED_x_corr_stack(2, x));

		bpX.store(std::move(x));

		return next ? next->forwardPropagation(vec(std::move(y))) : std::move(vec(std::move(y)));


	}
	virtual vec forwardPropagation_express(vec data) {
//std::cout << " fp " << std::endl;
		tensor x = data;
		x.reshape( { img_r, img_c, img_d });
		y = w.ADJUSTED_x_corr_stack(2, x);

		return next ? next->forwardPropagation_express(vec(std::move(y))) : std::move(vec(std::move(y)));

	}
	virtual vec backwardPropagation(vec error) {
		//std::cout << " backward prop adj conv" << std::endl;
		tensor dy = error;
		dy.reshape( { out_r, out_c, numb_krnls });
		tensor x(bpX.poll_last());
//std::cout << " non  lin meth " << std::endl;
			w_gradientStorage -= nonLin::adj_convolution_filter_error(w, x, dy);

		//	std::cout << " prev " <<std::endl;
		if (prev) {
			dx.zeros();
				dx += nonLin::adj_convolution_img_error(w,x, dy);


			//send it off into the world
			return prev->backwardPropagation(vec(dx));
		} else {
			return error;
		}
	}
	virtual vec backwardPropagation_ThroughTime(vec dy) {
		return dy;
	}

	//NeuralNetwork update-methods
	virtual void clearBackPropagationStorage() {
		bpX.clear();
	}
	virtual void clearGradientStorage() {
		w_gradientStorage.zeros();
	}
	void constrain(tensor& t) {
		for (unsigned i = 0; i < t.size(); ++i) {
			if (t.data()[i] < 0) {
				t.data()[i] = 0;
			}
			else if (t.data()[i] > 1) {
				t.data()[i] = 1;
			}
		}
	}
	virtual void updateGradients() {
		w += w_gradientStorage % lr;
		constrain(w);
	}
};
#endif

//depreacted code

//	-- This is the (non optimized) krnl for calculating the deltas/gradients of the current weights
//		unsigned e_id = 0;
//		for (unsigned f = 0; f < numb_krnls; ++f) {
//for each filter
//			for (unsigned r = 0; r < r_pos; ++r) {	//multiply the subtensor by the value
//				for (unsigned c = 0; c < c_pos; ++c) {
//					w_gradientStorage[f] -= x( { 0, r, c }, { krnl_r,krnl_c, img_d }) & dy(e_id);
//					++e_id;
//					e_id = e_id % dy.size();
//				}
//			}
//		}
//	-- non optimized code for calculating the gradients of dx (the error of the inputs)
//for (unsigned f = 0; f < numb_krnls; ++f) {
//	for (unsigned r = 0; r < r_pos; ++r) {
//		for (unsigned c = 0; c < c_pos; ++c) {
//			dx({0, r, c}, {krnl_r, krnl_c, img_d}) -= w[f] & dy(e_id);
//			++e_id;
//			e_id = e_id % dy.size();
//		}
//	}
//}

/*
 * Q_Learner.h
 *
 *  Created on: Oct 4, 2017
 *      Author: joseph
 */

#ifndef Q_LEARNER_H_
#define Q_LEARNER_H_

#include "Layer.h"

class Q_Learner: public Layer
{

	std::vector<int> max_indexes;

	virtual vec forwardPropagation(vec x)  {
		bpX.store(x);

		return x;
	}
	virtual vec forwardPropagation_express(vec x) {
		return x;
	}
	virtual vec backwardPropagation(vec dy) {

	}
	virtual vec backwardPropagation_ThroughTime(vec dy) = 0;

	//NeuralNetwork update-methods
	virtual void clearBackPropagationStorage() = 0;
	virtual void clearGradientStorage() = 0;
	virtual void updateGradients() = 0;

};

#endif /* Q_LEARNER_H_ */

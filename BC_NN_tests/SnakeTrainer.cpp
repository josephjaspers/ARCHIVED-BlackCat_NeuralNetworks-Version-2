//#include "SnakeGame.cpp"
//
//#include "NeuralNetwork.h"
//#include "FeedForward.h"
//#include "GatedRecurrentUnit.h"
//#include "Q_Learner.h"
//#include "ConvolutionalLayer.h"
//#include "BC_NeuralNetwork_Definitions.h"
//#include "LSTM.h"
//
//#include <iostream>
//#include <unistd.h>
//
//void snake_trainer() {
//
//	//set up the neural network for q learning
//
//	const unsigned ROWS = 10;
//	const unsigned COLS = 10;
//
//	std::cout << "initialize network " << std::endl;
//	NeuralNetwork net;
//	//ConvolutionalLayer c(ROWS, COLS, 1, 5, 5, 3);
//	GatedRecurrentUnit r(ROWS * COLS, 150);
//	FeedForward f(150, 4);
//
//
//	//f.setTanh();
//	//r.setTanh();
//
//	Q_Learner q(4);
//
//	net.add(&r);	//add convolutional layer
//	net.add(&f);	//add recurrent layer
//	//net.add(&f);	//add feedforward layer
//	net.add(&q);	//add q_learning layer
//
//	net.setLearningRate(.1);
//
//	std::cout << "initialize game " << std::endl;
//	//set up the snake game (pretty easy)
//	Snake game;
//	game.initGame(ROWS, COLS);
//
//
//	//game.playSnake(); //play the game yourself
//
//	unsigned games_played = 0;
//	const unsigned MAX_GAMES = 20000000;
//	const unsigned TRUNCATE_BP = 10;
//	//unsigned max_moves = 10;
//
//	typedef std::vector<Vector<double>> reward_storage;
//	reward_storage bp_rewards;
//	vec curr_reward(1);
//
//
//	while (games_played < MAX_GAMES) {
//
//		vec network_action = net.forwardPropagation(game.getBoardState());
//
//		value max = -2;
//		int mv = -2;
//		for (int i = 0; i < 4; ++i) {
//			if (network_action.data()[i] > max) {
//				max = network_action.data()[i];
//				mv = i;
//			}
//		}
//
//		game.move(mv);
//
//		double points = game.update();
//		curr_reward = points;
//
//		bp_rewards.push_back(curr_reward);
//
//		if (games_played % 300 == 0 && games_played != 0) {
//
//			q.setExploreRate(0);
//
//			game.print();
//			sleep(1);
//		} else {
//			q.setExploreRate(.7);
//		}
//
//		 if (curr_reward.data()[0] == -1) {
//			net.backwardPropagation(bp_rewards.back());
//			bp_rewards.pop_back();
//
//			unsigned curr_timestamp = 0;
//
//			while (!bp_rewards.empty() && curr_timestamp < TRUNCATE_BP) {
//				net.backwardPropagation_ThroughTime(bp_rewards.back());
//				bp_rewards.pop_back();
//				++curr_timestamp;
//			}
//
//			if (games_played % 300 == 0 && games_played != 0) {
//				std::cout << "points earned = " << game.getPoints() << std::endl;
//				std::cout << "high score    = " << game.getHighScore() << std::endl;
//				std::cout << "games played  = " << games_played + 1 << std::endl;
//			}
//			game.initGame();
//			games_played++;
//
//			net.update();
//			bp_rewards.clear();
//		}
//
//	}
//
//}
//
//int main123123() {
//	snake_trainer();
//	return 0;
//}

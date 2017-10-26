#include "Tensor.h"
#include "Matrix.h"

#ifndef BC_simple_Qtest
#define BC_simple_Qtest

class q_test {
public:
	unsigned consecutive_points = 0;
	unsigned highScore = 0;

	typedef std::pair<unsigned, unsigned> index;

	enum direction {
		up, down, left, right
	};

	Matrix<double> state;

	//the reward / value of action
	const double REWARD_VALUE = 1;
	const double PUNISH_VALUE = 0;
	const double ACTION_VALUE = 0.5;

	//represenation on the game state
	const double PLAYER_POSITION_VALUE = 1;
	const double REWARD_POSITION_VALUE = .6;
	const double PUNISH_POSITION_VALUE = .3;

	//positions
	index reward_position;
	index punish_position;
	index player_position;

	const unsigned ROWS;
	const unsigned COLS;

	q_test(unsigned r, unsigned c) :
			ROWS(r), COLS(c) {
		state = Matrix<double>(r, c);

		initGame();
	}

	void initGame() {
		player_position.first = rand() % ROWS;
		player_position.second = rand() % COLS;

		reward_position.first = rand() % ROWS;
		reward_position.second = rand() % COLS;
		while (player_position != reward_position) {
			reward_position.first = rand() % ROWS;
			reward_position.second = rand() % COLS;
		}

		punish_position.first = rand() % ROWS;
		punish_position.second = rand() % COLS;
		while (player_position != punish_position && punish_position != player_position) {
			punish_position.first = rand() % ROWS;
			punish_position.second = rand() % COLS;
		}
	}
	void print() {
		state.zeros();
		state[player_position.second][player_position.first] = PLAYER_POSITION_VALUE;
		state[reward_position.second][reward_position.first] = REWARD_POSITION_VALUE;
		state[punish_position.second][punish_position.first] = PUNISH_POSITION_VALUE;
	}
	Matrix<double> boardState() {
		state.zeros();
		state[player_position.second][player_position.first] = PLAYER_POSITION_VALUE;
		state[reward_position.second][reward_position.first] = REWARD_POSITION_VALUE;
		state[punish_position.second][punish_position.first] = PUNISH_POSITION_VALUE;

		return state;
	}

	double evaluateReward() {
		if (reward_position == player_position) {
			consecutive_points++;
			if (highScore < consecutive_points) {
				highScore = consecutive_points;
			}

			initGame();
			return REWARD_VALUE;
		} else if (punish_position == player_position) {
			initGame();
			consecutive_points = 0;
			return PUNISH_VALUE;
		} else {
			return ACTION_VALUE;
		}
	}

	double move(int mv) {
		return

		mv == 0 ? move(left) : mv == 1 ? move(right) : mv == 2 ? move(up) : move(down);
	}

	double move(direction mv) {
		if (mv == left) {
			if (player_position.second > 0) {
				player_position.second -= 1;
			}
		}
		if (mv == right) {
			if (player_position.second > COLS - 1) {
				player_position.second += 1;
			}
		}
		if (mv == up) {
			if (player_position.first > 0) {
				player_position.first -= 1;
			}
		}
		if (mv == down) {
			if (player_position.first < ROWS - 1) {
				player_position.first += 1;
			}
		}

		return evaluateReward();
	}
};

#endif

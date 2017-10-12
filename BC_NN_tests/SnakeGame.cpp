/*
 * Snake.h
 *
 *  Created on: Sep 27, 2017
 *      Author: joseph
 */

#ifndef SNAKE_H_
#define SNAKE_H_

#include <list>
#include "Tensor.h"
#include <stdio.h>
class Snake {


	unsigned points = 0; //also is the size of the snake minus the head
	unsigned highScore = 0;

	unsigned board_rows = 20;
	unsigned board_cols = 20;

	double FRUIT = .5;
	double HEAD = 1;
	double BODY = .8;

	Tensor<double> board;

	std::pair<unsigned, unsigned> fruit_position;
	std::pair<unsigned,unsigned> snake_head;
	std::list<std::pair<unsigned, unsigned>> snake_body;

public:
	enum direction {up, down, left, right};


	void initGame() {
		if (highScore < points) {
			highScore = points;
		}
		points = 0;
		board = Tensor<double>({board_rows, board_cols});


		snake_head.first = rand() % board_rows;
		snake_head.second = rand() % board_cols;

		fruit_position = snake_head;
		initFruit();
	}

	Snake() {
		initGame();
	}

	void print() {
		Tensor<double> p = {board_rows, board_cols}; p = 0;

		std::cout << "Head: " << snake_head.second << " " << snake_head.first << std::endl;
		std::cout << "Fruit: " << fruit_position.second << " " << fruit_position.first << std::endl;


		for (auto i = snake_body.begin(); i != snake_body.end(); ++i) {
			p((*i).first * board_rows + (*i).second) = BODY;
		}

		p(fruit_position.first * board_rows+fruit_position.second) = FRUIT;



		for (unsigned i = 0; i < board_cols+1; ++i) {
			std::cout << "#";
		}
		std::cout << std::endl;

		for (int i = 0; i < board_rows; ++i){
		std::cout << "#";
			for (int  j = 0; j < board_cols; ++j) {
				if (p[i](j)() == FRUIT) {
					std::cout << "F";
				} else if (p[i](j)() == HEAD) {
					std::cout << "O";
				} else if (p[i](j)() == BODY) {
					std::cout << "o";
				} else {
					std::cout << " ";
				}
			}
			std::cout << "#";
			std::cout << std::endl;
		}

		for (unsigned i = 0; i < board_cols+1; ++i) {
			std::cout << "#";
		}
		std::cout << std::endl;
	}

	int move(direction move) {

		auto tPoint = snake_body.back();

		if (points > 0) {
			auto tail = snake_body.back();
			snake_body.pop_back();
			tail = snake_head;
			snake_body.push_front(tail);
		}

		switch (move) {
		case up: snake_head.second += 1; break;
		case left: snake_head.first -= 1;break;
		case right: snake_head.first += 1;break;
		case down: snake_head.second -= 1; break;
		}

		if (fruit_position == snake_head) {
			snake_body.push_back(tPoint);
		}
	}
	bool validFruitPosition() {
		if (fruit_position == snake_head) {
			return false;
		}

		for (auto i = snake_body.begin(); i != snake_body.end(); ++i) {
			if ((*i) == fruit_position) {
				return false;
			}
		}
		return true;
	}
	void initFruit() {

			while (!validFruitPosition()) {
		fruit_position.first = rand() % board_rows;
		fruit_position.second = rand() % board_cols;

			}
	}

	int update() {

		if (fruit_position == snake_head) {
			++points;
			initFruit();
			return 1;
		}
		if (snake_head.first > board_rows) {
			initGame();
			return -1;
		}
		if (snake_head.second > board_cols) {
			initGame();
			return -1;
		} else {
			for (auto it = snake_body.begin(); it != snake_body.end(); ++it) {
				if (snake_head == (*it)) {
					initGame();
					return -1;
				}
			}
		}
		return 0;
	}
};


int main4() {
std::cout << " snake game " << std::endl;

	Snake game;

	while (game.update() != -1) {
		game.print();

		Snake::direction mv;
		int input;
		std::cin >> input;
		switch (input) {
		case 1: mv = Snake::direction::up; break;
		case 2: mv = Snake::direction::down; break;
		case 3: mv = Snake::direction::left; break;
		case 4: mv = Snake::direction::right; break;
		default: continue;
		}
		game.move(mv);
	}
	return 0;
}


#endif /* SNAKE_H_ */

#ifndef BLACKJACK_H
#define BLACKJACK_H

#include "EnvironmentsBase.hpp"
#include <random>
#include <cmath>
#include <utility>
#include <vector>
#include <algorithm>    

namespace octorl {

    class Blackjack : virtual public EnvironmentsBase {
        private: 
            static const int action_space_size = 2;
            static const int observation_space_size = 3;
            std::vector<int> player_hand;
            std::vector<int> dealer_hand;
            int deck[13]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10};
        public:
            Blackjack();

            StepReturn step(int action);
            StepReturn reset();
            std::vector<int> drawHand();
            bool usableAce(std::vector<int> hand);
            int rawSum(std::vector<int> hand);
            int sumHand(std::vector<int> hand);
            bool isBust(std::vector<int> hand);
            int score(std::vector<int> hand);
            bool isNatrual(std::vector<int> hand);
            int getActionSize();
            int getObservationSize();
            int currentStep();
            void playerView();
            int memorySize();
    };
}

#endif
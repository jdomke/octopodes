#include "../../include/envs/Blackjack.hpp"



octorl::Blackjack::Blackjack() {
    //srand()
    reset();
    
}

octorl::StepReturn octorl::Blackjack::step(int action) {
    octorl::StepReturn obs;
    if(action == 1) {
        player_hand.push_back(deck[rand()%13]);
        if(isBust(player_hand)) {
            obs.terminated = true;
            obs.done = true;
            obs.reward = -1;
        }
        
    }
    else {
        obs.terminated = true;
        obs.done = true;
        while(sumHand(dealer_hand) < 17)
            dealer_hand.push_back(deck[rand() % 13]);
        obs.reward = (float) (score(player_hand) > score(dealer_hand)) - (float) (score(player_hand) < score(dealer_hand));

        if(isNatrual(player_hand) && obs.reward == 1)
            obs.reward = 1.5;
        
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    int showing = dealer_hand[0];
    if(showing == 1)
        showing = 11;

    obs.observation =  torch::tensor({{sumHand(player_hand), dealer_hand[0],(int) usableAce(player_hand)}},options);
    return obs;
}

octorl::StepReturn octorl::Blackjack::reset() {
    player_hand = std::vector<int>{deck[rand() % 13],deck[rand() % 13]};//drawHand();
    dealer_hand = std::vector<int>{deck[rand() % 13],deck[rand() % 13]};//drawHand();
    octorl::StepReturn obs;
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    int showing = dealer_hand[0];
    if(showing == 1)
        showing = 11;

    obs.observation =  torch::tensor({{sumHand(player_hand), showing,(int) usableAce(player_hand)}},options);
    return obs;
}

std::vector<int> octorl::Blackjack::drawHand() {
    return std::vector<int>{deck[rand() % 13],deck[rand() % 13]};
}


bool octorl::Blackjack::usableAce(std::vector<int> hand) {
    for(auto i : hand) {
        if(i == 1)
            return rawSum(hand) + 10 <= 21;
    }
    return false;
}


int octorl::Blackjack::rawSum(std::vector<int> hand) {
    int sum = 0;
    for(auto i : hand)
        sum += i;
    return sum;
}

int octorl::Blackjack::sumHand(std::vector<int> hand) {
    if(usableAce(hand))
        return rawSum(hand) + 10;
    return rawSum(hand); 
}

bool octorl::Blackjack::isBust(std::vector<int> hand) {
    return sumHand(hand) > 21;
}

int octorl::Blackjack::score(std::vector<int> hand) {
    if(isBust(hand))
        return 0;
    return sumHand(hand);
}

bool octorl::Blackjack::isNatrual(std::vector<int> hand) {
    if(hand.size() == 2 && (hand[0] == 1 && hand[1] == 10) || (hand[0] == 10 && hand[1] == 1))
        return true;
    return false;

}

int octorl::Blackjack::getActionSize() {
    return action_space_size;
}

int octorl::Blackjack::getObservationSize() {
    return observation_space_size;
}

int octorl::Blackjack::currentStep() {
    return 0;
}

void octorl::Blackjack::playerView() {


}
#include "../include/quickTest.hpp"

using namespace std;
void cartpoleCheckTest() {

    fstream fin;
    octorl::StepReturn obs;
    shared_ptr<octorl::Cartpole> env(new octorl::Cartpole());

    fin.open("../environment_test_data/cartpoleTest.csv", ios::in);

    vector<string> row;
    string line, word, temp;
    float x, x_dot, theta, theta_dot, x1, x_dot1, theta1, theta_dot1;
    int action;
    int fails = 0;
    int total = 0;
    getline(fin, line);
    while(!fin.eof()) {
        total++;
        row.clear();

        getline(fin, line);
        if(fin.eof())
            break;
        
        stringstream s(line);
  
        while (getline(s, word, ',')) {
            row.push_back(word);
        }
        x = stof(row[0]);
        x_dot = stof(row[1]);
        theta = stof(row[2]);
        theta_dot = stof(row[3]);
        //cout<<x<<x_dot<<theta<<theta_dot<<endl;
        action = stoi(row[4]);
        x1 = stof(row[5]);
        x_dot1 = stof(row[6]);
        theta1 = stof(row[7]);
        theta_dot1 = stof(row[8]);
        
        env->setState(x,x_dot,theta,theta_dot);
        obs = env->step(action);
        // cout<<endl;
        // cout<<obs.observation<<endl;
        // cout<<x1<<" "<<x_dot1<<" "<<theta1<<" "<<theta_dot1<<endl;
        // cout<<endl;
        bool check = ((obs.observation[0][0] - x1).item().toDouble() < pow(10,-3)) && 
        ((obs.observation[0][1] - x_dot1).item().toDouble() < pow(10,-3)) &&
        ((obs.observation[0][2] - theta1).item().toDouble() < pow(10,-3)) &&
        ((obs.observation[0][3] - theta_dot1).item().toDouble() < pow(10,-3));

        if(!check){
            cout<<obs.observation<<endl;
            cout<<x1<<x_dot1<<theta1<<theta_dot1<<endl;
            cout<<((obs.observation[0][0] - x1))<<endl;
            cout<<((obs.observation[0][1] - x_dot1))<<endl;
            cout<<((obs.observation[0][2] - theta1))<<endl;
            cout<<((obs.observation[0][3] - theta_dot1))<<endl;
            cout<<"fail\n";

            fails++;
            //return;
        }
//        return;
    }
    cout<<"Fails: "<<(double)fails/(double)total<<endl;
}



void mountainCarCheckTest() {

    fstream fin;
    octorl::StepReturn obs;
    shared_ptr<octorl::MountainCar> env(new octorl::MountainCar());

    fin.open("../environment_test_data/MountainCarTest.csv", ios::in);

    vector<string> row;
    string line, word, temp;


    double position, velocity, position1, velocity1;
    int action;
    int fails = 0;
    int total = 0;
    getline(fin, line);
    //cout<<line<<endl;
    //fin.get(); 
    while(!fin.eof()) {

        total++;
        row.clear();

        getline(fin, line);
        //cout<<line<<endl;
        //fin.get(); 
        if(fin.eof())
            break;
        stringstream s(line);
  
        while (getline(s, word, ',')) {
            row.push_back(word);
        }
        
        // cout<<total<<endl;
        // cout<<line<<endl;
        // cout<<row[0]<<endl;
        position = stod(row[0]);
        
       // cout<<row[1]<<endl;
        velocity = stod(row[1]);
       // cout<<row[2]<<endl;
        action = stoi(row[2]);
        //cout<<row[3]<<endl;
        position1 = stod(row[3]);
        //cout<<row[4]<<endl;
        velocity1 = stod(row[4]);
        env->setState(position, velocity);
        obs = env->step(action);

    //  cout<<endl;
    //     cout<<row[0]<<endl;
    //     cout<<position<<" "<<velocity<<endl;
    //     cout<<obs.observation[0][0]<<" "<<obs.observation[0][1]<<endl;
    //     cout<<position1<<" "<<velocity1<<endl;
    //     cout<<obs.observation - torch::tensor({{position1, velocity1}})<<endl;
    //     cout<<endl;
         bool check = ((obs.observation[0][0].item().toDouble() - position1) < pow(10,-2)) && 
        ((obs.observation[0][1].item().toDouble() - velocity1) < pow(10,-2));

        if(!check){
            
            cout<<obs.observation[0][0].item().toDouble()<<" "<<position1<<endl;
            cout<<((obs.observation[0][0].item().toDouble() - position1))<<endl;
            cout<<obs.observation[0][1].item().toDouble()<<" "<<velocity1<<endl;
            cout<<((obs.observation[0][1].item().toDouble() - velocity1))<<endl;
            cout<<"fail\n";

            fails++;
        }
    }
    cout<<"Fails: "<<(double)fails/(double)total<<endl;
}
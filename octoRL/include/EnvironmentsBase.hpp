#ifndef ENVIRONMENTS_BASE_H
#define ENVIRONMENTS_BASE_H
#include "Spaces.hpp"


namespace octorl {

    class EnvironmentsBase {
        public:
            virtual StepReturn step(int action) = 0;
        private:
        protected:

    };
}

#endif
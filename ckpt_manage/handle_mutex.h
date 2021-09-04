#ifndef HANDLE_M_
#define HANDLE_M_

#include <pybind11/pybind11.h>
#include <pthread.h>
#include <semaphore.h>

namespace py = pybind11;

class HandleMutex
{
    private:
        sem_t updted, ckpted;
        
    public:
        HandleMutex();
        ~HandleMutex();
        py::bool_ TRY_P_CKPT();
        void P_CKPT();
        void V_CKPT();
        void P_UPDT();
        void V_UPDT();
};

#endif
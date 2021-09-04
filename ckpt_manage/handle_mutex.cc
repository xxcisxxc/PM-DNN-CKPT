#include "handle_mutex.h"

HandleMutex::HandleMutex()
{
    sem_init(&updted, 0, 0);
    sem_init(&ckpted, 0, 1);
}

HandleMutex::~HandleMutex()
{
    sem_close(&updted);
    sem_close(&ckpted);
}

py::bool_ HandleMutex::TRY_P_CKPT()
{
    bool isckpted = false;
    if (sem_trywait(&ckpted) != 0)
    {
        isckpted = true;
    }
    else
    {
        V_CKPT();
    }
    return isckpted;
}

void HandleMutex::P_CKPT()
{
    sem_wait(&ckpted);
}

void HandleMutex::P_UPDT()
{
    sem_wait(&updted);
}

void HandleMutex::V_CKPT()
{
    sem_post(&ckpted);
}

void HandleMutex::V_UPDT()
{
    sem_post(&updted);
}

PYBIND11_MODULE(handle_mutex, m) {
    py::class_<HandleMutex>(m, "HandleMutex")
        .def(py::init<>())
        .def("TRY_P_CKPT", &HandleMutex::TRY_P_CKPT)
        .def("P_CKPT", &HandleMutex::P_CKPT)
        .def("P_UPDT", &HandleMutex::P_UPDT)
        .def("V_CKPT", &HandleMutex::V_CKPT)
        .def("V_UPDT", &HandleMutex::V_UPDT);
}
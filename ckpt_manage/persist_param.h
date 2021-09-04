#ifndef PERSIST_PM_H_
#define PERSIST_PM_H_

#include <pybind11/pybind11.h>
#include "torch/extension.h"
#include <libpmemobj.h>

#include <string>
#include <unordered_map>
#include <atomic>
#include <vector>

#include "handle_mutex.h"

namespace py = pybind11;
using namespace pybind11::literals;

#define CHAR_BYTES sizeof(char)
#define FLOAT_BYTES sizeof(float)
#define PY_SSIZE_BYTES sizeof(Py_ssize_t)
#define DATA_HEAD_BYTES sizeof(DataHead)
#define LAYOUT_NAME "PERSIST_PARAM"

inline Py_ssize_t NBytesM(Py_ssize_t a, Py_ssize_t m)
{
    return (Py_ssize_t)(a / m) + (Py_ssize_t)((bool)(a % m));
}

inline Py_ssize_t NMBytes(Py_ssize_t a, Py_ssize_t m)
{
    return NBytesM(a, m) * m;
}

inline Py_ssize_t POOL_SIZE(Py_ssize_t a)
{
    return a + PMEMOBJ_MIN_POOL;
}

struct DataHead
{
    unsigned short nlen_name; // the length of a parameter name
    unsigned short itemsize; // the byte numbers of an item
    unsigned short dtype; // the type of an item
    unsigned short ndim; // the dim of a parameter
    Py_ssize_t num_tensor; // the total items of a tensor
};

struct Data
{
    DataHead info; // the info of a parameter
    Py_ssize_t is_psst; // whether the data has already persisted
    /***
     * char param_name []
     * ssize_t sizes []
     * ssize_t strides []
     * void tensor []
    ***/
    char value[1]; // concrete value of a parameter
};

struct BaseAddrInfo
{
    std::atomic<char *> addr;
    std::atomic<Py_ssize_t> cur_pos;
    std::atomic<Py_ssize_t> offset;

    BaseAddrInfo() : addr(nullptr), cur_pos(0), offset(0) {}
    inline BaseAddrInfo(BaseAddrInfo &&BAI)
    {
        addr = BAI.addr.load();
        cur_pos = BAI.cur_pos.load();
        offset = BAI.offset.load();
    }
    inline void set(char *addr_, Py_ssize_t cur_pos_, Py_ssize_t offset_)
    {
        addr = addr_;
        cur_pos = cur_pos_;
        offset = offset_;
    }
};

class PersistParam
{
    private:
        Data *persist_data;
        PMEMobjpool *pop;

        std::unordered_map<std::string, Py_ssize_t> name_offset;
        std::unordered_map<std::string, BaseAddrInfo> baseaddr;

        void _create_persist(const char *filename, py::list names, py::list values);
        void _open_persist(const char *filename);
        void _set_fromall(std::string name, char *oldtensor, Py_ssize_t nbytes, Py_ssize_t *flag);
        void _set_frombase(std::string name, char *oldtensor, Py_ssize_t nbytes, Py_ssize_t *flag);

    public:
        PersistParam(py::str filename, py::list names=py::list(), py::list values=py::list());
        ~PersistParam();
        torch::Tensor getvalue(py::str name);
        void _setvalue(Py_ssize_t offset, void *c_tensor, HandleMutex *handler_c, bool change);
        py::tuple getall();
        void change_base(py::str name);
        void create_setval_thread(py::str name, torch::Tensor obj_tensor, py::object handler_py, py::bool_ change);
};

struct Args_Set
{
    PersistParam *pparam;
    Py_ssize_t offset;
    void *c_tensor;
    HandleMutex *handler_c;
    bool change;
};

void * _setvalue_thread(void *args_setval);

py::bool_ complete_ckpted();

void AllThreadsEnd();

#endif
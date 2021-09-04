#include "persist_param.h"

std::atomic<int> count_threads(0);

void PersistParam::_create_persist(const char *filename, py::list names, py::list values)
{
    /***
     * First, Calculate the space we need.
     * Note that this space will be unchanged.
    ***/
    
    /* Initialize some variables */
    Py_ssize_t num_params = names.size(); /* Number of parameters we have */
    Py_ssize_t cur_pos = 0; /* Record the current cursor */
    Py_ssize_t len_name, num_dim, nbytes_tensor; /* size in Data::value */
    std::string hash_key; /* This hash table to store positions, volatile */
    torch::Tensor obj_tensor; /* extracted tensor */
    
    /* Start Calculating Space */
    for (Py_ssize_t i = 0; i < num_params; i++)
    {
        /* Add (name, pos) to Hash Table */
        hash_key = PyUnicode_AsUTF8AndSize(names[i].ptr(), &len_name);
        name_offset[hash_key] = cur_pos;
        baseaddr[hash_key];

        cur_pos += DATA_HEAD_BYTES; /* Space for DataHead */
        cur_pos += PY_SSIZE_BYTES; /* Space for is_psst */
        cur_pos += NMBytes(CHAR_BYTES * (++len_name), 8); /* Space for Parameter Name */
        
        /* Calculate Space for Size & Stride */
        obj_tensor = py::cast<torch::Tensor>(values[i]);
        cur_pos += (PY_SSIZE_BYTES * obj_tensor.dim() * 2);

        /* Calculate Space for Data */
        cur_pos += NMBytes(obj_tensor.nbytes(), 8);
    }

    /***
     * open or create the persistent pool
    ***/
    remove(filename);

    pop = pmemobj_create(filename, LAYOUT_NAME, POOL_SIZE(++cur_pos), 0666);
    PMEMoid root = pmemobj_root(pop, cur_pos);
	persist_data = (Data *)pmemobj_direct(root);

    /***
     * Second, Put the initial value into the space.
    ***/
    
    const char *name; /* returned UTF-8 name of parameter */
    Py_ssize_t nbytes_2s; /* number of bytes of sizes and strides */
    Data *visit_data = persist_data;

    /* Data Pointers to the content of persist_data */
    char *name_val;
    Py_ssize_t *shape_val, *strides_val;

    /* Start Copying */
    for (Py_ssize_t i = 0; i < num_params; i++)
    {
        cur_pos = 0; /* pointer to the entrance of Data::value */

        visit_data->is_psst = 0;
        pmemobj_persist(pop, &(visit_data->is_psst), PY_SSIZE_BYTES);

        /* Get the Name and Put into Data::value */
        name = PyUnicode_AsUTF8AndSize(names[i].ptr(), &len_name);
        name_val = visit_data->value;
        pmemobj_memcpy_persist(pop, name_val, name, ++len_name);

        len_name = NMBytes(CHAR_BYTES * len_name, 8);
        (visit_data->info).nlen_name = len_name; /* Update the Head::name */

        cur_pos += len_name; /* move to the next item */
        
        obj_tensor = py::cast<torch::Tensor>(values[i]);

        /* Update the Head::format */
        (visit_data->info).dtype = torch::utils::aten_to_numpy_dtype(obj_tensor.scalar_type());

        /* Update the Head::ndim */
        num_dim = obj_tensor.dim();
        (visit_data->info).ndim = num_dim;

        /* Pointer to the Data::value::size */
        shape_val = (Py_ssize_t *)(name_val + cur_pos);

        nbytes_2s = PY_SSIZE_BYTES * num_dim; /* size or stride length */

        /* Pointer to the Data::value::stride */
        cur_pos += nbytes_2s;
        strides_val = (Py_ssize_t *)(name_val + cur_pos);
        
        /* Update Size & Stride */
        for (Py_ssize_t j = 0; j < num_dim; j++)
        {
            shape_val[j] = obj_tensor.size(j);
            strides_val[j] = obj_tensor.stride(j);
        }

        pmemobj_persist(pop, shape_val, 2*nbytes_2s);

        cur_pos += nbytes_2s; /* move to the next item */

        /* Copy Data */
        (visit_data->info).num_tensor = obj_tensor.numel();
        (visit_data->info).itemsize = obj_tensor.itemsize();
        pmemobj_persist(pop, &(visit_data->info), DATA_HEAD_BYTES);

        nbytes_tensor = obj_tensor.nbytes();
        pmemobj_memcpy_persist(pop, name_val+cur_pos, obj_tensor.data_ptr(), nbytes_tensor);

        visit_data->is_psst = 1;
        pmemobj_persist(pop, &(visit_data->is_psst), PY_SSIZE_BYTES);

        cur_pos += NMBytes(nbytes_tensor, 8); /* move to the next item */

        /* Move visit_data to the next Data */
        visit_data = (Data *)(name_val + cur_pos);
    }
    *(char *)visit_data = '\0';
    pmemobj_persist(pop, visit_data, CHAR_BYTES);
}

void PersistParam::_open_persist(const char *filename)
{
    pop = pmemobj_open(filename, LAYOUT_NAME);
    PMEMoid root = pmemobj_root(pop, pmemobj_root_size(pop));
	persist_data = (Data *)pmemobj_direct(root);
    
    Data *visit_data = persist_data;
    std::string hash_key;
    Py_ssize_t cur_pos = 0, offset;

    while (*(char *)visit_data != '\0')
    {
        DataHead &OInfo = visit_data->info;

        /* Add (name, pos) to Hash Table */
        hash_key = std::string(visit_data->value);
        name_offset[hash_key] = cur_pos;
        baseaddr[hash_key];

        cur_pos += DATA_HEAD_BYTES; /* Space for DataHead */
        cur_pos += PY_SSIZE_BYTES; /* Space for is_psst */

        offset = OInfo.nlen_name; /* Calculate Space for Parameter Name */
        
        /* Calculate Space for Size & Stride */
        offset += (PY_SSIZE_BYTES * OInfo.ndim * 2);

        /* Calculate Space for Data */
        offset += NMBytes(OInfo.itemsize * OInfo.num_tensor, 8);
        visit_data = (Data *)(visit_data->value+offset);

        cur_pos += offset; /* Space for value */
    }
}

PersistParam::PersistParam(py::str filename, py::list names, py::list values)
{
    if (names.empty())
        _open_persist(std::string(filename).data());
    else
        _create_persist(std::string(filename).data(), names, values);
}

PersistParam::~PersistParam()
{
    pmemobj_close(pop);
}

torch::Tensor PersistParam::getvalue(py::str name)
{
    /* Find the entrance of "name"'s data */
    Py_ssize_t offset = name_offset[name];
    Data *data = (Data *)((char *)persist_data + offset);
    DataHead &OInfo = data->info;

    /* Find the entrance of "name"'s sizes, strides */
    offset = OInfo.nlen_name * CHAR_BYTES;
    
    /* get sizes and strides */
    Py_ssize_t *shape_strides = (Py_ssize_t *)(data->value + offset);
    Py_ssize_t num_dim = OInfo.ndim;
    std::vector<Py_ssize_t> sizes(shape_strides, shape_strides+num_dim);
    shape_strides += num_dim;
    std::vector<Py_ssize_t> strides(shape_strides, shape_strides+num_dim);

    /* Find the entrance of "name"'s tensor */
    Py_ssize_t nbytes_2s = num_dim * PY_SSIZE_BYTES * 2;
    offset += nbytes_2s;

    /* Copy tensor to new tensor */
    Py_ssize_t nbytes_tensor = OInfo.num_tensor * OInfo.itemsize;
    void *c_tensor = malloc(nbytes_tensor);
    memcpy(c_tensor, data->value + offset, nbytes_tensor);

    /* Get the dtype */
    at::ScalarType dt = torch::utils::numpy_dtype_to_aten((data->info).dtype);

    return torch::from_blob(c_tensor, sizes, strides, torch::dtype(dt));
}

void PersistParam::create_setval_thread(py::str name, torch::Tensor obj_tensor, py::object handler_py, py::bool_ change)
{
    count_threads++;

    void *c_tensor = obj_tensor.data_ptr();
    HandleMutex *handler_c = handler_py.cast<HandleMutex *>();
    bool change_ = (bool)change;
    Py_ssize_t offset = name_offset[name];
    
    baseaddr.at(name).set((char *)c_tensor, 0, (Py_ssize_t)change_-1);

    py::gil_scoped_release release;

    Args_Set *args_setval = new Args_Set;
    *args_setval = {this, offset, c_tensor, handler_c, change_};

    pthread_t setval_thread;
    pthread_create(&setval_thread, NULL, _setvalue_thread, args_setval);

    py::gil_scoped_acquire acquire;
}

void * _setvalue_thread(void *args_setval)
{
    pthread_detach(pthread_self());

    Args_Set *args = (Args_Set *)args_setval;
    args->pparam->_setvalue(args->offset, args->c_tensor, args->handler_c, args->change);

    delete args;
    count_threads--;
    pthread_exit(0);
}

void PersistParam::_setvalue(Py_ssize_t offset, void *c_tensor, HandleMutex *handler_c, bool change)
{
    Data *data = (Data *)((char *)persist_data + offset);

    /* Calculate tensor information */
    //bool change = true;
    std::string name(data->value);
    DataHead &OInfo = data->info;
    offset = OInfo.nlen_name * CHAR_BYTES + 2 * OInfo.ndim * PY_SSIZE_BYTES;
    Py_ssize_t nbytes_tensor = OInfo.itemsize * OInfo.num_tensor;

    /* Start Checkpoint */
    handler_c->P_UPDT();

    /* Flag to ensure consistency */
    data->is_psst = 0;
    pmemobj_persist(pop, &(data->is_psst), PY_SSIZE_BYTES);

    if (change)
        _set_frombase(name, data->value+offset, nbytes_tensor, &(data->is_psst));
    else
        _set_fromall(name, data->value+offset, nbytes_tensor, &(data->is_psst));
    
    /* End Checkpoint */
    handler_c->V_CKPT();
}

py::tuple PersistParam::getall()
{
    py::list names, values;
    Data *visit_data = persist_data;
    Py_ssize_t offset, nbytes_tensor, nbytes_2s;
    Py_ssize_t *shape_strides;
    while (*(char *)visit_data != '\0')
    {
        py::str name = py::str(visit_data->value);
        names.append(name);

        DataHead &OInfo = visit_data->info;
        offset = OInfo.nlen_name * CHAR_BYTES;
        nbytes_2s = 2 * OInfo.ndim * PY_SSIZE_BYTES;
        nbytes_tensor = OInfo.itemsize * OInfo.num_tensor;

        shape_strides = (Py_ssize_t *)(visit_data->value + offset);
        std::vector<Py_ssize_t> sizes(shape_strides, shape_strides+OInfo.ndim);
        shape_strides += OInfo.ndim;
        std::vector<Py_ssize_t> strides(shape_strides, shape_strides+OInfo.ndim);

        offset += nbytes_2s;

        void *c_tensor = malloc(nbytes_tensor);
        memcpy(c_tensor, visit_data->value + offset, nbytes_tensor);

        at::ScalarType dt = torch::utils::numpy_dtype_to_aten(OInfo.dtype);
        values.append(torch::from_blob(c_tensor, sizes, strides, torch::dtype(dt)));

        offset += NMBytes(nbytes_tensor, 8);
        visit_data = (Data *)(visit_data->value+offset);
    }
    return py::make_tuple(names, values);
}

py::bool_ complete_ckpted()
{
    if (count_threads == 0)
        return true;
    return false;
}

void AllThreadsEnd()
{
    while (count_threads != 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("AllThreadEnd", &AllThreadsEnd);
    m.def("complete_ckpted", &complete_ckpted);
    py::class_<PersistParam>(m, "PersistParam")
        .def(py::init<py::str, py::list, py::list>(), "filename"_a, "names"_a = py::list(), "values"_a = py::list())
        .def("getvalue", &PersistParam::getvalue, "name"_a)
        .def("create_setval_thread", &PersistParam::create_setval_thread, "name"_a, "obj_tensor"_a, "handler_py"_a, "change"_a)
        .def("getall", &PersistParam::getall)
        .def("change_base", &PersistParam::change_base, "name"_a);
}
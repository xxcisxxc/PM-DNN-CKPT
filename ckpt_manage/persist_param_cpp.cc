#include "persist_param.h"

void PersistParam::_set_fromall(std::string name, char *oldtensor, Py_ssize_t nbytes, Py_ssize_t *flag)
{
    BaseAddrInfo &BAInfo = baseaddr.at(name);
    const char *newtensor = BAInfo.addr;
    memcpy(oldtensor, newtensor, nbytes);
    BAInfo.offset = -2;
    pmemobj_persist(pop, oldtensor, nbytes);

    *flag = 1;
    pmemobj_persist(pop, flag, PY_SSIZE_BYTES);
}

void PersistParam::_set_frombase(std::string name, char *oldtensor, Py_ssize_t nbytes, Py_ssize_t *flag)
{
    BaseAddrInfo &BAInfo = baseaddr.at(name);

    BAInfo.cur_pos = nbytes % 8;
    memcpy(oldtensor, BAInfo.addr, BAInfo.cur_pos);
    for (; BAInfo.cur_pos < nbytes; BAInfo.cur_pos += 8)
    {
        memcpy(oldtensor+BAInfo.cur_pos, BAInfo.addr+BAInfo.cur_pos, 8);
    }
    pmemobj_persist(pop, oldtensor, nbytes);

    *flag = 1;
    pmemobj_persist(pop, flag, PY_SSIZE_BYTES);

    if (BAInfo.offset > 0)
        free(BAInfo.addr+BAInfo.offset);
}

void PersistParam::change_base(py::str name)
{
    BaseAddrInfo &BAInfo = baseaddr.at(name);
    if (BAInfo.offset < 0)
    {
        while (BAInfo.offset != -2);
        return;
    }

    Py_ssize_t offset = name_offset[name];
    Data *data = (Data *)((char *)persist_data + offset);
    DataHead &OInfo = data->info;
    Py_ssize_t nbytes = OInfo.itemsize * OInfo.num_tensor;

    const char *c_tensor = (const char *)BAInfo.addr;
    Py_ssize_t rev_cur = nbytes;

    char *part_tensor;

    offset = BAInfo.cur_pos;
    if (offset < nbytes)
        part_tensor = (char *)malloc(nbytes-offset) - offset;
    else
        return;
    Py_ssize_t cur_offset = BAInfo.cur_pos;
    while (cur_offset < nbytes && cur_offset < rev_cur)
    {
        memcpy(part_tensor+rev_cur-8, c_tensor+rev_cur-8, 8);
        rev_cur -= 8;
        cur_offset = BAInfo.cur_pos;
    }
    
    if (BAInfo.cur_pos < nbytes)
    {
        BAInfo.addr = part_tensor;
        BAInfo.offset = offset;
    }
    else
        free(part_tensor+offset);
}
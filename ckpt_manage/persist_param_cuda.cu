#include "persist_param.h"

#include <cuda.h>
#include <cuda_runtime.h>

void cudaInfo(int *optimal_size, int *lprior, int *hprior, Py_ssize_t nbytes)
{
    const int max_streams = 100;
    cudaDeviceGetStreamPriorityRange(lprior, hprior);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, device);
    *optimal_size = devProp.maxThreadsPerBlock;
    if (nbytes > (*optimal_size * max_streams))
    {
        *optimal_size = (int)(nbytes / (8 * max_streams)) * 8;
    }
}

struct StreamInfo
{
    cudaStream_t stream;
    Py_ssize_t offset;
    Py_ssize_t length;
};

struct CoWStreamInfo
{
    cudaStream_t stream;
    char *part_tensor;
};

void PersistParam::_set_fromall(std::string name, char *oldtensor, Py_ssize_t nbytes, Py_ssize_t *flag)
{
    BaseAddrInfo &BAInfo = baseaddr.at(name);
    const char *newtensor = BAInfo.addr;
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostRegister(oldtensor, nbytes, cudaHostRegisterMapped);

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cudaMemcpyAsync(oldtensor, newtensor, nbytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    BAInfo.cur_pos = 2;

    cudaHostUnregister(oldtensor);
    //cudaMemcpy(oldtensor, newtensor, nbytes, cudaMemcpyDeviceToHost);

    pmemobj_persist(pop, oldtensor, nbytes);

    *flag = 1;
    pmemobj_persist(pop, flag, PY_SSIZE_BYTES);
}

void PersistParam::_set_frombase(std::string name, char *oldtensor, Py_ssize_t nbytes, Py_ssize_t *flag)
{
    int optimal_size, lprior, hprior;
    cudaInfo(&optimal_size, &lprior, &hprior, nbytes);
    
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostRegister(oldtensor, nbytes, cudaHostRegisterMapped);

    Py_ssize_t num_streams = NBytesM(nbytes, optimal_size);
    std::vector<StreamInfo> copy_streams(num_streams);
    
    BaseAddrInfo &BAInfo = baseaddr.at(name);
    BAInfo.cur_pos = 0;
    char *addr = BAInfo.addr;
    Py_ssize_t cur_pos = nbytes % optimal_size;
    if (cur_pos == 0) cur_pos = optimal_size;

    cudaStreamCreateWithFlags(&(copy_streams[0].stream), cudaStreamNonBlocking);
    cudaMemcpyAsync(oldtensor, addr, cur_pos, cudaMemcpyDeviceToHost, copy_streams[0].stream);
    copy_streams[0].offset = 0;
    copy_streams[0].length = cur_pos;
    for (Py_ssize_t i = 1; i < num_streams; cur_pos += optimal_size, i++)
    {
        cudaStreamCreateWithFlags(&(copy_streams[0].stream), cudaStreamNonBlocking);
        cudaMemcpyAsync(oldtensor+cur_pos, addr+cur_pos, optimal_size, cudaMemcpyDeviceToHost, copy_streams[i].stream);
        copy_streams[i].offset = cur_pos;
        copy_streams[i].length = optimal_size;
    }

    std::vector<CoWStreamInfo> CoW_streams;
    std::vector<StreamInfo> Remain_streams;
    while (!copy_streams.empty())
    {
        Py_ssize_t i = 0;
        while (i < copy_streams.size())
        {
            if (cudaStreamQuery(copy_streams[i].stream) == cudaSuccess)
            {
                cudaStreamDestroy(copy_streams[i].stream);
                copy_streams.erase(copy_streams.begin() + i);
                continue;
            }
            else
            {
                if (BAInfo.cur_pos == 1)
                {
                    char *part_tensor;
                    cudaMalloc(&part_tensor, copy_streams[i].length);

                    cudaStream_t highprior_stream;
                    cudaStreamCreateWithPriority(&highprior_stream, cudaStreamNonBlocking, hprior);
                    cudaMemcpyAsync(part_tensor, addr+copy_streams[i].offset, copy_streams[i].length, cudaMemcpyDeviceToDevice, highprior_stream);

                    CoW_streams.push_back({highprior_stream, part_tensor});
                    cudaMemcpyAsync(oldtensor+copy_streams[i].offset, part_tensor, copy_streams[i].length, cudaMemcpyDeviceToHost, copy_streams[i].stream);

                    Remain_streams.push_back(copy_streams[i]);
                    copy_streams.erase(copy_streams.begin() + i);
                }
                else
                    i += 1;
            }
        }
    }

    for (CoWStreamInfo &CoWS : CoW_streams)
    {
        cudaStreamSynchronize(CoWS.stream);
        cudaStreamDestroy(CoWS.stream);
    }
    BAInfo.cur_pos = 2;
    
    for (Py_ssize_t i = 0; i < Remain_streams.size(); i++)
    {
        cudaStreamSynchronize(Remain_streams[i].stream);
        cudaStreamDestroy(Remain_streams[i].stream);
        cudaFree(CoW_streams[i].part_tensor);
    }

    cudaHostUnregister(oldtensor);

    pmemobj_persist(pop, oldtensor, nbytes);

    *flag = 1;
    pmemobj_persist(pop, flag, PY_SSIZE_BYTES);
}

void PersistParam::change_base(py::str name)
{
    BaseAddrInfo &BAInfo = baseaddr.at(name);

    if (BAInfo.cur_pos == 2)
    {
        return;
    }
    BAInfo.cur_pos = 1;
    while (BAInfo.cur_pos != 2);
}
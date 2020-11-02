#include "openpifpaf/tensorrt_net.hpp"
#include "openpifpaf/trt_utils.hpp"
#include "openpifpaf/upsample.hpp"
#include <torch/torch.h>
namespace F = torch::nn::functional;

inline void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    NV_CUDA_CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

TensorRTNet::TensorRTNet(const uint batchSize, const NetworkInfo& networkInfo):
    mEnginePath(networkInfo.enginePath),
    mConfigFilePath(networkInfo.configFilePath),
    mInputH(0),
    mInputW(0),
    mInputC(0),
    mInputSize(0),
    mUpscaleHeatmapFactor(2),
    mCmapChannels(0),
    mPafChannels(0),
    mMaxPerson(0),
    mCmapThreshold(0.1),
    mLinkThreshold(0.05),
    mWindowSize(9),
    mTopology(nullptr),
    mPrintPerfInfo(true),
    mLogger(Logger()),
    mBatchSize(batchSize),
    mEngine(nullptr),
    mContext(nullptr),
    mInputBindingIndex(-1),
    mTrtRunTime(nullptr)
{
    this->parseConfigFile();
    int outHeight = mUpscaleHeatmapFactor*mOutputH;
    int outWidth = mUpscaleHeatmapFactor*mOutputW;
    mKeypointParser = new KeypointParser(outHeight, outWidth, mBatchSize,mTopology,
                                          mCmapChannels, mPafChannels, mMaxPerson,
                                          mCmapThreshold, mLinkThreshold, mWindowSize,
                                          mNumLinesIntegral, mUpscaleHeatmapFactor);

    this->loadEngine();
    assert(mEngine != nullptr);
    mContext = mEngine->createExecutionContext();
    assert(mContext != nullptr);
    assert(mBatchSize <= static_cast<uint>(mEngine->getMaxBatchSize()));
    getBindingIdxs();
    allocateBuffers();
    NV_CUDA_CHECK(cudaStreamCreate(&mStream));

    assert(verifyTensorRTNetEngine());

}

void TensorRTNet::loadEngine(){
        std::fstream file;
        file.open(mEnginePath, ios::binary | ios::in);
        if(!file.is_open())
        {
            std::cout << "read engine file" << mEnginePath <<" failed" << std::endl;
            return;
        }
        file.seekg(0, ios::end);
        int length = file.tellg();
        file.seekg(0, ios::beg);
        std::unique_ptr<char[]> data(new char[length]);
        file.read(data.get(), length);

        file.close();

        std::cout << "deserializing" << std::endl;
        mTrtRunTime = nvinfer1::createInferRuntime(mLogger);
        assert(mTrtRunTime != nullptr);
        mEngine = mTrtRunTime->deserializeCudaEngine(data.get(), length, NULL);
        assert(mEngine != nullptr);
}


void TensorRTNet::doInference()
{
    int volumindex[2] = {mCmapChannels, mPafChannels};
    mContext->enqueue(mBatchSize, mDeviceBuffers.data(), mStream, nullptr);

    for (auto& tensor : mOutputTensors)
    {
        int binding_index = tensor.bindingIndex;
        if(mUpscaleHeatmapFactor == 1)
        {
            NV_CUDA_CHECK(cudaMemcpyAsync(tensor.hostBuffer, mDeviceBuffers.at(binding_index),
                                      mBatchSize * tensor.volume * sizeof(float),
                                      cudaMemcpyDeviceToHost, mStream));
        }
        else
        {
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0).requires_grad(false);
            std::vector<int64_t> sizes = {mBatchSize, volumindex[binding_index-1], mOutputH, mOutputW};
            auto out_tensor = torch::fromblob(mDeviceBuffers.at(binding_index), at::IntList(sizes), options=options);
            auto upsample_tensor = F::interpolate(out_tensor, F::InterpolateFuncOptions()
                                                                .size(std::vector<int64_t>{mUpscaleHeatmapFactor*mOutputH, mUpscaleHeatmapFactor*mOutputW})
                                                                .mode(torch::kBicubic).align_corners(false));
            NV_CUDA_CHECK(cudaMemcpyAsync(tensor.hostBuffer,
                                          upsample_tensor.data_ptr(),
                                          mBatchSize * tensor.volume * sizeof(float),
                                          cudaMemcpyDeviceToHost, mStream));
        }
    }
    cudaStreamSynchronize(mStream);
}

void TensorRTNet::allocateBuffers()
{

    mDeviceBuffers.resize(mEngine->getNbBindings(), nullptr);
    mOutputTensors.resize(mEngine->getNbBindings() - mNumberInput, nullptr);

    assert(mInputBindingIndex != -1 && "Invalid input binding index");

    // allocate buffers for input buffers (gpu Mem)
    nvinfer1::Dims input_dims = mEngine->getBindingDimensions(mInputBindingIndex);
    nvinfer1::DataType input_dtype = mEngine->getBindingDataType(mInputBindingIndex);
    int64_t input_totalSize = volume(input_dims) * mBatchSize * getElementSize(input_dtype);
    safeCudaMalloc(&mDeviceBuffers.at(mInputBindingIndex), input_totalSize);

    // allocate Buffers for device outputs and host outputs
    for (auto output_idx: mOutputBindingIndexes)
    {
        int hostOutputIdx = output_idx - mNumberInput;
        nvinfer1::Dims dim = mEngine->getBindingDimensions(output_idx);
        nvinfer1::DataType dtype = mEngine->getBindingDataType(output_idx);
        int64_t totalSize = volume(dim) * mBatchSize * getElementSize(dtype);
        safeCudaMalloc(&mDeviceBuffers.at(output_idx), totalSize);
        mOutputTensors[hostOutputIdx].volume = totalSize;
        mOutputTensors[hostOutputIdx].bindingIndex = output_idx;
        NV_CUDA_CHECK(cudaMallocHost(&(mOutputTensors[hostOutputIdx].hostBuffer),
                                        totalSize * mBatchSize));
    }
}

void TensorRTNet::getBindingIndxes()
{

    int numbindings_per_profile = mEngine->getNbBindings() / mEngine->getNbOptimizationProfiles();
    int start_binding = 0;
    int end_binding = start_binding + numbindings_per_profile;
    for (int binding_index = start_binding; binding_index < end_binding; binding_index++)
    {
        if(mEngine->bindingIsInput(binding_index))
            {
                mNumberInput ++;
                mInputBindingIndex = binding_index;
            }
        else
            mOutputBindingIndexes.push_back(binding_index);
    }

}

bool TensorRTNet::verifyTensorRTNetEngine()
{
    assert((mEngine->getNbBindings() == (1 + mOutputTensors.size())
            && "Binding info doesn't match between cfg and engine file \n"));

    assert(mEngine->bindingIsInput(mInputBindingIndex) && "Incorrect input binding index \n");
    assert(get3DTensorVolume(mEngine->getBindingDimensions(mInputBindingIndex)) == mInputSize);
    return true;
}

TensorRTNet::~TensorRTNet()
{
    for (auto& tensor : mOutputTensors) NV_CUDA_CHECK(cudaFreeHost(tensor.hostBuffer));
    for (auto& deviceBuffer : mDeviceBuffers) NV_CUDA_CHECK(cudaFree(deviceBuffer));
    if (mContext)
    {
        mContext->destroy();
        mContext = nullptr;
    }
    if(mTrtRunTime)
    {
        mTrtRunTime->destroy();
        mTrtRunTime = nullptr;
    }
    if (mEngine)
    {
        mEngine->destroy();
        mEngine = nullptr;
    }
    cudaStreamSynchronize(mStream);
    cudaStreamDestroy(mStream);

}

void TensorRTNet::parseConfigFile()
{
    nlohmann::json jsonConfig = readJsonFile(mConfigFilePath);
    // net info
    mInputH = jsonConfig["net_info"]["input_h"];
    mInputW = jsonConfig["net_info"]["input_w"];
    mInputC = jsonConfig["net_info"]["input_c"];
    mOutputH = jsonConfig["net_info"]["output_h"];
    mOutputW = jsonConfig["net_info"]["output_w"];
    // input volum
    mInputSize = mInputH * mInputW * mInputC;
    // output parser info
}
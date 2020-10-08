/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "openPifPafNet.h"
#include "tensorConvert.h"

#include "utils/cuda/cudaMappedMemory.h"
#include "utils/cuda/cudaResize.h"

#include "utils/commandLine.h"
#include "utils/filesystem.h"
#include "utils/logging.h"


// constructor
PifpafNet::PifpafNet() : tensorNet()
{
	mNetworkType   = CUSTOM;
}


// destructor
PifpafNet::~PifpafNet()
{

}


// NetworkTypeFromStr
PifpafNet::NetworkType PifpafNet::NetworkTypeFromStr( const char* modelName )
{
	if( !modelName )
		return PifpafNet::CUSTOM;

	PifpafNet::NetworkType type = PifpafNet::CUSTOM;

	if( strcasecmp(modelName, "resnet-50") == 0 || strcasecmp(modelName, "resnet_50") == 0 || strcasecmp(modelName, "resnet50") == 0 )
		type = PifpafNet::RESNET_50;

	return type;
}


// NetworkTypeToStr
const char* PifpafNet::NetworkTypeToStr( PifpafNet::NetworkType network )
{
	switch(network)
	{
		case PifpafNet::RESNET_50:	return "ResNet-50";
	}

	return "Custom";
}

// PreProcess
bool PifpafNet::PreProcess( void* image, uint32_t width, uint32_t height, imageFormat format )
{
	// verify parameters
	if( !image || width == 0 || height == 0 )
	{
		LogError(LOG_TRT "PifpafNet::PreProcess( 0x%p, %u, %u ) -> invalid parameters\n", image, width, height);
		return false;
	}

	if( !imageFormatIsRGB(format) )
	{
		LogError(LOG_TRT "PifpafNet::Classify() -- unsupported image format (%s)\n", imageFormatToStr(format));
		LogError(LOG_TRT "                        supported formats are:\n");
		LogError(LOG_TRT "                           * rgb8\n");
		LogError(LOG_TRT "                           * rgba8\n");
		LogError(LOG_TRT "                           * rgb32f\n");
		LogError(LOG_TRT "                           * rgba32f\n");

		return false;
	}

	PROFILER_BEGIN(PROFILER_PREPROCESS);
	// downsample, convert to band-sequential RGB, and apply pixel normalization, mean pixel subtraction and standard deviation
	if( CUDA_FAILED(cudaTensorNormMeanRGB(image, format, width, height, 
									mInputs[0].CUDA, GetInputWidth(), GetInputHeight(), 
									make_float2(0.0f, 1.0f), 
									make_float3(0.485f, 0.456f, 0.406f),
									make_float3(0.229f, 0.224f, 0.225f), 
									GetStream())) )
	{
		LogError(LOG_TRT "PifpafNet::PreProcess() -- cudaTensorNormMeanRGB() failed\n");
		return false;
	}

	PROFILER_END(PROFILER_PREPROCESS);
	return true;
}


// Process
bool PifpafNet::Process()
{
	PROFILER_BEGIN(PROFILER_NETWORK);

	if( !ProcessNetwork() )
		return false;

	PROFILER_END(PROFILER_NETWORK);
	return true;
}


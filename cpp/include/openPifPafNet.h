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

#ifndef __OPEN_PIFPAF_NET_H__
#define __OPEN_PIFPAF_NET_H__


#include "tensorNet.h"


/**
 * Name of default input blob for PifpafNet model.
 * @ingroup PifpafNet
 */
#define PIFPAFNET_DEFAULT_INPUT   "input"

/**
 * Name of default output confidence values for pifpafNet model.
 * @ingroup PifpafNet
 */
#define PIFPAFNET_DEFAULT_PIF  "pif"

/**
 * Name of default output confidence values for pifpafNet model.
 * @ingroup PifpafNet
 */
#define PIFPAFNET_DEFAULT_PAF  "paf"

/**
 * Standard command-line options able to be passed to PifpafNet::Create()
 * @ingroup PifpafNet
 */
#define PIFPAFNET_USAGE_STRING  "PifpafNet arguments: \n" 							\
		  "  --network=NETWORK    pre-trained model to load, one of the following:\n" 	\
		  "                           * resnet-50\n" 							\
		  "  --model=MODEL        path to custom model to load (caffemodel, uff, or onnx)\n" 			\
		  "  --prototxt=PROTOTXT  path to custom prototxt to load (for .caffemodel only)\n" 				\
		  "  --input-blob=INPUT   name of the input layer (default is '" PIFPAFNET_DEFAULT_INPUT "')\n" 	\
		  "  --output-pif=PIF  name of the pif output layer (default is '" PIFPAFNET_DEFAULT_PIF "')\n" 	\
		  "  --output-paf=PAF  name of the paf output layer (default is '" PIFPAFNET_DEFAULT_PAF "')\n" 	\
		  "  --batch-size=BATCH   maximum batch size (default is 1)\n"								\
		  "  --profile            enable layer profiling in TensorRT\n\n"


/**
 * Image recognition with classification networks, using TensorRT.
 * @ingroup PifpafNet
 */
class PifpafNet : public tensorNet
{
public:
	/**
	 * Network choice enumeration.
	 */
	enum NetworkType
	{
		CUSTOM,        /**< Custom model provided by the user */
		RESNET_50,	/**< ResNet-50 backbone*/
	};

	/**
	 * Parse a string to one of the built-in pretrained models.
	 * Valid names are "alexnet", "googlenet", "googlenet-12", or "googlenet_12", ect.
	 * @returns one of the PifpafNet::NetworkType enums, or PifpafNet::CUSTOM on invalid string.
	 */
	static NetworkType NetworkTypeFromStr( const char* model_name );

	/**
	 * Convert a NetworkType enum to a string.
	 */
	static const char* NetworkTypeToStr( NetworkType network );

	/**
	 * Load a new network instance by parsing the command line.
	 */
	static PifpafNet* Create( int argc, char** argv );

	/**
	 * Load a new network instance by parsing the command line.
	 */
	static PifpafNet* Create( const commandLine& cmdLine );

	/**
	 * Usage string for command line arguments to Create()
	 */
	static inline const char* Usage() 		{ return PIFPAFNET_USAGE_STRING; }

	/**
	 * Destroy
	 */
	virtual ~PifpafNet();

	/**
	 * Retrieve the network type (alexnet or googlenet)
	 */
	inline NetworkType GetNetworkType() const					{ return mNetworkType; }

	/**
 	 * Retrieve a string describing the network name.
	 */
	inline const char* GetNetworkName() const					{ NetworkTypeToStr(mNetworkType); }

protected:
	PifpafNet();

	bool PreProcess( void* image, uint32_t width, uint32_t height, imageFormat format );
	bool Process();
	NetworkType mNetworkType;
};


#endif

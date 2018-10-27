/**
 * This file is part of the feature compression framework used in:
 * "Collaborative Visual SLAM using Compressed Feature Exchange"
 *
 * Copyright (C) 2017-2018 Dominik Van Opdenbosch <dominik dot van-opdenbosch at tum dot de>
 * Chair of Media Technology, Technical University of Munich
 * For more information see <https://d-vo.github.io/>
 *
 * The feature compression framework is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * The feature compression framework is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with the feature compression framework. If not, see <http://www.gnu.org/licenses/>.
 */

#include "feature_coder.h"
#include <chrono>
#include <omp.h>

#define MAX_NUM_FEATURES 2500

//#define ANDROID
//#define NO_DEPTH_Q


namespace LBFC2
{

long long FeatureCoder::imageId = 0;

FeatureCoder::FeatureCoder(ORBVocabulary &voc, CodingStats &model, int imWidth, int imHeight, int maxOctave, int angleBins,
		int bufferSize, bool inter, bool stereo, bool depth, float focalLength, float baseline, int threads)
: mModel(model), mVoc(voc), mImWidth(imWidth), mImHeight(imHeight), mLevels(maxOctave), mAngleBins(angleBins),
  mBufferSize(bufferSize), mbInterPred(inter), mbStereo(stereo), mbMonoDepth(depth), mfFocalLength(focalLength),
  mfBaseline(baseline)
{
	mbCurrentViewRight = false;


	mAngleBinSize = 360.0 / mAngleBins;

	// INTRA CODING
	mFreqIntraRes.resize(2);
	mFreqIntraRes[0] = (int)max(1, (int)round( mModel.p0_intra_ * (double)AC_PRECISION ) );
	mFreqIntraRes[1] = AC_PRECISION - mFreqIntraRes[0];




	// Pyramid sizes for intra coding
	mvPyramidWidth.resize(mModel.mOctaves);
	mvPyramidHeight.resize(mModel.mOctaves);
	mvPyramidWidth[0] = mImWidth;
	mvPyramidHeight[0] = mImHeight;


	// Pyramid bits for intra coding
	mvBitsPyramidWidth.resize(mModel.mOctaves);
	mvBitsPyramidHeight.resize(mModel.mOctaves);
	mvBitsPyramidWidth[0] = log2(mvPyramidWidth[0]);
	mvBitsPyramidHeight[0] = log2(mvPyramidHeight[0]);

	mScaleFactors.resize(mModel.mOctaves);
	mScaleFactors[0] = 1.0;

	for( int d = 1; d < mModel.mOctaves; d++ )
	{
		// Pyramid sizes
		mScaleFactors[d] = mScaleFactors[d-1]*mModel.mScaleFactor;
		mvPyramidWidth[d] = ceil(((float) mImWidth) / mScaleFactors[d]);
		mvPyramidHeight[d] = ceil(((float) mImHeight) / mScaleFactors[d]);

		// Pyramid bits
		mvBitsPyramidWidth[d] = log2(mvPyramidWidth[d]);
		mvBitsPyramidHeight[d] = log2(mvPyramidHeight[d]);
	}


	// Intra keypoint costs
	mnBitsOctave = log2(mLevels);
	mnBitsAngle = log2(mAngleBins);
	mnBitsBow = log2(mVoc.size());

	mnAngleOffset = mModel.mAngleBins-1;
	mnOctaveOffset =  mModel.mOctaves-1;


	// Intra bow propabilities
	size_t voc_size = voc.size();
	mFreqIntraBow.resize(voc_size);
	for( size_t i = 0; i < voc_size; i++ )
	{
		const double prob = 1.0 / voc_size;
		mFreqIntraBow[i] = (int)max(1, (int)round( prob * (double)AC_PRECISION ) );
	}


	// Intra keypoint propabilities
	mFreqIntraPosX.resize(mModel.mOctaves);
	mFreqIntraPosY.resize(mModel.mOctaves);
	for( int d = 0; d < mModel.mOctaves; d++ )
	{
		mFreqIntraPosX[d] = cv::Mat(1, mvPyramidWidth[d], CV_32S, cv::Scalar::all(1 ));
		mFreqIntraPosY[d] = cv::Mat(1, mvPyramidHeight[d], CV_32S, cv::Scalar::all(1 ));
	}

	mFreqIntraOctave.resize(mLevels);
	std::fill(mFreqIntraOctave.begin(), mFreqIntraOctave.end(), 1);

	mFreqIntraAngle.resize(mAngleBins);
	std::fill(mFreqIntraAngle.begin(), mFreqIntraAngle.end(), 1);


	// INTER CODING
	// Inter residual coding propabilities
	mFreqInterRes.resize(2);
	mFreqInterRes[0] = (int)max(1, (int)round( mModel.p0_inter_ * (double)AC_PRECISION ) );
	mFreqInterRes[1] = AC_PRECISION - mFreqInterRes[0];


	mFreqInterAngleDiff.resize(mModel.pAngleDelta_.cols);
	for( int d = 0; d < mModel.pAngleDelta_.cols; d++ )
	{
		const float &prob = mModel.pAngleDelta_.at<float>(d);
		mFreqInterAngleDiff[d] = (int)max(1, (int)round( prob * (double)AC_PRECISION ) );
	}


	mFreqInterOctaveDiff.resize(mModel.pOctaveDelta_.cols);
	for( int d = 0; d < mModel.pOctaveDelta_.cols; d++ )
	{
		float prob = mModel.pOctaveDelta_.at<float>(d);
		mFreqInterOctaveDiff[d] = (int)max(1, (int)round( prob * (double)AC_PRECISION ) );
	}


	// Inter coding propabilities
	mFreqInterKeyPoint.resize(mModel.mOctaves);
	mPInterKeyPoint.resize(mModel.mOctaves);

	for( int d = 0; d < mModel.mOctaves; d++ )
	{
		mPInterKeyPoint[d] = mModel.pPosPerOctave_[d];
		mFreqInterKeyPoint[d] = cv::Mat(mPInterKeyPoint[d].rows, mPInterKeyPoint[d].cols, CV_32S, cv::Scalar::all(0));
		for( int x = 0; x <  mModel.pPosPerOctave_[d].cols; x++)
		{
			for( int y = 0; y <  mModel.pPosPerOctave_[d].rows; y++)
			{
				float prob = mPInterKeyPoint[d].at<float>(y,x);
				mFreqInterKeyPoint[d].at<int>(y,x) = (int)max(1, (int)round( prob * (double)AC_PRECISION ) );
			}
		}
	}


	// STEREO CODING
	// Stereo residual coding propabilities
	mFreqStereoRes.resize(2);
	mFreqStereoRes[0] = (int)max(1, (int)round( mModel.p0_stereo_ * (double)AC_PRECISION ) );
	mFreqStereoRes[1] = AC_PRECISION - mFreqStereoRes[0];

	mfBaseLineFocalLength = mfBaseline * mfFocalLength;

	mFreqStereoPosX.resize(mModel.mOctaves);
	mFreqStereoPosY.resize(mModel.mOctaves);
	for( int d = 0; d < mModel.mOctaves; d++ )
	{
		mFreqStereoPosX[d] = cv::Mat(1, mvPyramidWidth[d], CV_32S, cv::Scalar::all(1 ));
		mFreqStereoPosY[d] = cv::Mat(1, 5, CV_32S, cv::Scalar::all(1));
	}


	mvSearchRangeStereoPyramid.resize(mModel.mOctaves);
	mvSearchRangeStereoPyramid[0] = mModel.mSearchRangeStereoX;
	mSearchRangeStereoY = mModel.mSearchRangeStereoY;
	for( int d = 1; d < mModel.mOctaves; d++ )
	{
		mvSearchRangeStereoPyramid[d] = ceil((float) mModel.mSearchRangeStereoX / mScaleFactors[d]);
	}



	// DEPTH CODING
	if( !mModel.mDepthCodebook.empty() )
	{
		for( int i = 0; i < mModel.mDepthCodebook.rows; i++ )
			mvfDepthCodeBook.push_back(mModel.mDepthCodebook.at<float>(i));
	}

	mnDepthBits = ceil(log2(mvfDepthCodeBook.size()));


	// Cost lookup tables
	mLutRIntra.resize(257);
	mLutRInter.resize(257);
	for( int d = 0; d < 257; d++ )
	{
		mLutRIntra[d] =  -((float)(256-d)) * log2(mModel.p0_intra_) - ((float) d) * log2(1.0 - mModel.p0_intra_);
		mLutRInter[d] =  -((float)(256-d)) * log2(mModel.p0_inter_) - ((float) d) * log2(1.0 - mModel.p0_inter_);
	}



	// Prepare coder
	mCurrentImageId = 0;
	nThreads = threads;
	vEncodeContext.resize(nThreads);
	vGlobalACCoderContext.resize(nThreads);

	for( int i = 0; i < nThreads; i++ )
		initEncoderModels(vGlobalACCoderContext[i]);

	vDecodeContext.resize(nThreads);
	vGlobalACDecoderContext.resize(nThreads);

	for( int i = 0; i < nThreads; i++ )
		initDecoderModels(vGlobalACDecoderContext[i]);
}



void FeatureCoder::encodeImage( std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descriptors, vector<uchar> &bitstream )
{
	cv:: Mat descriptorsRight;
	std::vector<cv::KeyPoint> kptsRight;
	return encodeImageStereo(keypoints, descriptors, kptsRight, descriptorsRight, bitstream );
}


void FeatureCoder::decodeImage( const vector<uchar> &bitstream, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors,
		std::vector<unsigned int> &visualWords )
{
	cv:: Mat descriptorsRight;
	std::vector<cv::KeyPoint> kptsRight;
	return decodeImageStereo( bitstream, keypoints, descriptors, kptsRight, descriptorsRight, visualWords );
}



void FeatureCoder::encodeImageDepth( const std::vector<cv::KeyPoint> &kptsLeft, const cv::Mat &descriptorsLeft,
		const std::vector<cv::KeyPoint> &kptsRight, const cv::Mat &descriptorsRight, vector<uchar> &bitstream  )
{
	// Just pass through
	return encodeImageStereo(kptsLeft, descriptorsLeft, kptsRight, descriptorsRight, bitstream );
}


void FeatureCoder::decodeImageDepth( const vector<uchar> &bitstream, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors,
		std::vector<unsigned int> &visualWords, std::vector<float> &vfDepthValues )
{
	cv:: Mat descriptorsRight;
	std::vector<cv::KeyPoint> kptsRight;
	decodeImageStereo( bitstream, keypoints, descriptors, kptsRight, descriptorsRight, visualWords );


	vfDepthValues = vector<float>(keypoints.size(),-1.0);
	for( size_t k = 0; k < keypoints.size(); k++ )
		vfDepthValues[k] = keypoints[k].size;
}




void FeatureCoder::encodeImageStereo( const std::vector<cv::KeyPoint> &kptsLeft, const cv::Mat &descriptorsLeft,
		const std::vector<cv::KeyPoint> &kptsRight, const cv::Mat &descriptorsRight, vector<uchar> &bitstream )
{
	bitstream.clear();


	// Left view
	mbCurrentViewRight = false;
	mCurrentImageBuffer = ImgBufferEntry(mImWidth, mImHeight, mLevels);
	mCurrentImageBuffer.mnImageId = imageId++;



	// Calculate matching
	std::vector<int> maskMatch1(kptsLeft.size(), -1);
	std::vector<int> maskMatch2(kptsRight.size(), -1);
	std::vector<float> vfDepthValues(kptsLeft.size(), -1.0);

	if( mbStereo || mbMonoDepth )
		matchStereoFeatures(kptsLeft, descriptorsLeft, kptsRight, descriptorsRight, maskMatch1, maskMatch2, vfDepthValues);


	// Pre-calculate decisions for left view
	std::vector<ModeDecision> decisionsLeft(kptsLeft.size());

#pragma omp parallel for
	for( size_t i = 0; i < kptsLeft.size(); i++ )
	{
		decisionsLeft[i] = modeDecision(kptsLeft[i], descriptorsLeft.row(i));
		decisionsLeft[i].keypointId = i;
		decisionsLeft[i].stereoMatch = maskMatch1[i] >= 0;
	}

	size_t numFeatures = 0;
	for( size_t i = 0; i < decisionsLeft.size(); i++ )
	{
		decisionsLeft[i].bufferIndex = numFeatures;
		numFeatures++;
	}


	int featPerThread = ceil(((float) numFeatures) / nThreads);
	std::vector<std::vector<unsigned int> > vLeftThreadDec(nThreads);
	for( size_t i = 0; i < decisionsLeft.size(); i++ )
	{
		ModeDecision &decision = decisionsLeft[i];
		int bufferIndex = decision.bufferIndex;
		int threadId = floor(((float) bufferIndex)  / featPerThread);
		vLeftThreadDec[threadId].push_back(i);
	}


	mCurrentImageBuffer.allocateSpace(numFeatures);
	std::vector<int> vLutKeyPointToBuffer(decisionsLeft.size(), -1);

	// Encode
#pragma omp parallel for
	for( int threadId = 0; threadId < nThreads; threadId++)
	{
		EncodeContext &globalCoderContext = vEncodeContext[threadId];
		ACEncodeContext &globalACCoderContext = vGlobalACCoderContext[threadId];
		for( size_t j = 0; j < vLeftThreadDec[threadId].size(); j++ )
		{
			size_t i = vLeftThreadDec[threadId][j];
			const ModeDecision &decision = decisionsLeft[i];

			// Let's go
			const int &kptId = decision.keypointId;
			const cv::KeyPoint &keypoint = kptsLeft[kptId];
			const cv::Mat &descriptor = descriptorsLeft.row(kptId);
			const float &fDepth = vfDepthValues[kptId];
			encodeFeature(decision, keypoint, descriptor, fDepth, globalCoderContext, globalACCoderContext);
			vLutKeyPointToBuffer[kptId] = decision.bufferIndex;
		}
	}
	const int nCodedLeft = numFeatures;

	mCurrentImageBuffer.AssignFeatures();
	mLeftImageBuffer.push_back(mCurrentImageBuffer);


	size_t numFeaturesRight = 0;

	if( !mbMonoDepth )
	{
		// Right view
		mbCurrentViewRight = true;
		mCurrentImageBuffer = ImgBufferEntry(mImWidth, mImHeight, mLevels);
		mCurrentImageBuffer.mnImageId = imageId++;



		// Pre-calculate decisions for right view
		std::vector<ModeDecision> decisionsRight(kptsRight.size());

#pragma omp parallel for
		for( size_t i = 0; i < kptsRight.size(); i++ )
		{
			int stereoMatchBufferId = -1;
			if( (maskMatch2[i] >= 0) )
			{
				stereoMatchBufferId = vLutKeyPointToBuffer[maskMatch2[i]];
			}

			decisionsRight[i] = modeDecision(kptsRight[i], descriptorsRight.row(i), stereoMatchBufferId);
			decisionsRight[i].keypointId = i;
			decisionsRight[i].stereoMatch = stereoMatchBufferId >= 0;
		}



		for( size_t i = 0; i < decisionsRight.size(); i++ )
		{
			decisionsRight[i].bufferIndex = numFeaturesRight;
			numFeaturesRight++;
		}

		int featPerThreadRight = ceil(((float) numFeaturesRight) / nThreads);
		std::vector<std::vector<unsigned int> > vRightThreadDec(nThreads);
		for( size_t i = 0; i < decisionsRight.size(); i++ )
		{
			ModeDecision &decision = decisionsRight[i];
			int bufferIndex = decision.bufferIndex;
			int threadId = floor(((float) bufferIndex)  / featPerThreadRight);
			vRightThreadDec[threadId].push_back(i);
		}

		mCurrentImageBuffer.allocateSpace(numFeaturesRight);

#pragma omp parallel for
		for( int threadId = 0; threadId < nThreads; threadId++)
		{
			// Let's go
			EncodeContext &globalCoderContext = vEncodeContext[threadId];
			ACEncodeContext &globalACCoderContext = vGlobalACCoderContext[threadId];
			for( size_t j = 0; j < vRightThreadDec[threadId].size(); j++ )
			{
				size_t i = vRightThreadDec[threadId][j];
				const ModeDecision &decision = decisionsRight[i];

				// Let's go
				const int &kptId = decision.keypointId;
				const cv::KeyPoint &keypoint = kptsRight[kptId];
				const cv::Mat &descriptor = descriptorsRight.row(kptId);
				const float fDepth = -1.0;
				encodeFeature(decision, keypoint, descriptor, fDepth, globalCoderContext, globalACCoderContext);
			}
		}



		mCurrentImageBuffer.AssignFeatures();
		mRightImageBuffer.push_back(mCurrentImageBuffer);
	}


	const int nCodedRight = numFeaturesRight;

	// Pop after right view
	if( mLeftImageBuffer.size() > mBufferSize)
		mLeftImageBuffer.pop_front();

	if( mRightImageBuffer.size() > mBufferSize)
		mRightImageBuffer.pop_front();


	for( int threadId = 0; threadId < nThreads; threadId++)
	{
		vEncodeContext[threadId].finish();
		vGlobalACCoderContext[threadId].finish();
	}


	bitstream.resize(sizeof(EncodeInfo));
	EncodeInfo *info = (EncodeInfo *) &bitstream[0];
	info->numFeatures = nCodedLeft;
	info->numFeaturesRight = nCodedRight;
	info->fixedLengthSize = nThreads;



	for( int threadId = 0; threadId < nThreads; threadId++)
	{
		int offset = bitstream.size();
		bitstream.resize(offset + sizeof(ThreadInfo));
		ThreadInfo *info = (ThreadInfo *) &bitstream[offset];
		info->fixedLengthSize = vEncodeContext[threadId].bitstream.size();
		info->variableLengthSize = vGlobalACCoderContext[threadId].bitstream.size();
	}

	for( int threadId = 0; threadId < nThreads; threadId++)
	{
		bitstream.insert(bitstream.end(), vEncodeContext[threadId].bitstream.begin(), vEncodeContext[threadId].bitstream.end());
		bitstream.insert(bitstream.end(), vGlobalACCoderContext[threadId].bitstream.begin(), vGlobalACCoderContext[threadId].bitstream.end());
	}

	for( int threadId = 0; threadId < nThreads; threadId++)
	{
		vEncodeContext[threadId].clear();
		vGlobalACCoderContext[threadId].clear();
	}

	mCurrentImageId++;
}



void FeatureCoder::decodeImageStereo( const vector<uchar> &bitstream, std::vector<cv::KeyPoint> &kptsLeft, cv::Mat &descriptorsLeft,
		std::vector<cv::KeyPoint> &kptsRight, cv::Mat &descriptorsRight, std::vector<unsigned int> &visualWords )
{
	EncodeInfo *info = (EncodeInfo *) &bitstream[0];
	const unsigned int nCodedLeft = info->numFeatures;
	const unsigned int nCodedRight = info->numFeaturesRight;
	const unsigned int numThreads = info->fixedLengthSize;


	size_t offset = sizeof(EncodeInfo);

	std::vector<ThreadInfo> vInfo;
	for( unsigned int threadId = 0; threadId < numThreads; threadId++)
	{
		ThreadInfo *info = (ThreadInfo *) &bitstream[offset];
		offset += sizeof(ThreadInfo);
		vInfo.push_back(*info);
	}


	vector<vector<uchar> > vvFLBitstream(numThreads);
	vector<list<uchar> > vAcBitstream(numThreads);


	std::vector<size_t> vOffsets = {offset};
	for( unsigned int threadId = 0; threadId < numThreads; threadId++)
	{
		size_t fixedLengthOffset = vInfo[threadId].fixedLengthSize;
		size_t variableLengthOffset = vInfo[threadId].variableLengthSize;
		offset += fixedLengthOffset + variableLengthOffset;
		vOffsets.push_back(offset);
	}

#pragma omp parallel for
	for( unsigned int threadId = 0; threadId < numThreads; threadId++)
	{
		const size_t offset = vOffsets[threadId];
		size_t fixedLengthOffset = vInfo[threadId].fixedLengthSize;
		size_t variableLengthOffset = vInfo[threadId].variableLengthSize;
		vvFLBitstream[threadId].assign(bitstream.begin() + offset, bitstream.begin() + offset + fixedLengthOffset);
		vAcBitstream[threadId].assign(bitstream.begin() + offset + fixedLengthOffset, bitstream.begin() + offset + fixedLengthOffset + variableLengthOffset);
	}


	for(unsigned int i = 0; i < numThreads; i++ )
	{
		vGlobalACDecoderContext[i].setBitstream(vAcBitstream[i]);
		vDecodeContext[i].clear();
		vDecodeContext[i].bitstream = vvFLBitstream[i];
	}


	// Left view
	mbCurrentViewRight = false;
	mCurrentImageBuffer = ImgBufferEntry(mImWidth, mImHeight, mLevels );
	mCurrentImageBuffer.allocateSpace(nCodedLeft);

	int featPerThreadLeft = ceil(((float) nCodedLeft) / numThreads);
	std::vector<std::vector<unsigned int> > vLeftFeatureIndex(numThreads);
	for( unsigned int i = 0; i < nCodedLeft; i++ )
	{
		int threadId = floor(((float) i) / featPerThreadLeft);
		vLeftFeatureIndex[threadId].push_back(i);
	}


#pragma omp parallel for
	for( unsigned int threadId = 0; threadId < numThreads; threadId++)
	{
		// Let's go
		DecodeContext &globalDecoderContext = vDecodeContext[threadId];
		ACDecodeContext &globalACDecoderContext = vGlobalACDecoderContext[threadId];
		for( size_t j = 0; j < vLeftFeatureIndex[threadId].size(); j++ )
		{
			size_t i = vLeftFeatureIndex[threadId][j];
			unsigned int visualWord = 0;
			cv::KeyPoint keypoint;
			cv::Mat descriptor;
			decodeFeature(keypoint, descriptor, visualWord, globalDecoderContext, globalACDecoderContext);
			mCurrentImageBuffer.addFeature(i, keypoint, descriptor, visualWord);
		}
	}

	kptsLeft = mCurrentImageBuffer.mvKeypoints;
	descriptorsLeft = mCurrentImageBuffer.mDescriptors;

	mCurrentImageBuffer.AssignFeatures();
	mLeftImageBuffer.push_back(mCurrentImageBuffer);



	// Right view
	mbCurrentViewRight = true;

	int featPerThreadRight = ceil(((float) nCodedRight) / numThreads);
	std::vector<std::vector<unsigned int> > vRightFeatureIndex(numThreads);
	for( unsigned int i = 0; i < nCodedRight; i++ )
	{
		int threadId = floor(((float) i) / featPerThreadRight);
		vRightFeatureIndex[threadId].push_back(i);
	}

	mCurrentImageBuffer = ImgBufferEntry(mImWidth, mImHeight, mLevels );
	mCurrentImageBuffer.allocateSpace(nCodedRight);

#pragma omp parallel for
	for( unsigned int threadId = 0; threadId < numThreads; threadId++)
	{
		// Let's go
		DecodeContext &globalDecoderContext = vDecodeContext[threadId];
		ACDecodeContext &globalACDecoderContext = vGlobalACDecoderContext[threadId];
		for( unsigned int j = 0; j < vRightFeatureIndex[threadId].size(); j++ )
		{
			size_t i = vRightFeatureIndex[threadId][j];
			unsigned int visualWord = 0;
			cv::KeyPoint keypoint;
			cv::Mat descriptor;
			decodeFeature(keypoint, descriptor, visualWord, globalDecoderContext, globalACDecoderContext);
			mCurrentImageBuffer.addFeature(i, keypoint, descriptor, visualWord);
		}
	}

	kptsRight = mCurrentImageBuffer.mvKeypoints;
	descriptorsRight = mCurrentImageBuffer.mDescriptors;


	mCurrentImageBuffer.AssignFeatures();
	mRightImageBuffer.push_back(mCurrentImageBuffer);


	if( mLeftImageBuffer.size() > mBufferSize)
		mLeftImageBuffer.pop_front();

	if( mRightImageBuffer.size() > mBufferSize)
		mRightImageBuffer.pop_front();
}



unsigned int FeatureCoder::encodeFeature(const ModeDecision &decision, const cv::KeyPoint &keypoint,
		const cv::Mat &descriptor, const float &fDepth, EncodeContext &globalCoderContext,
		ACEncodeContext &globalACCoderContext )
{
	const unsigned int bits_start = globalCoderContext.bits() + globalACCoderContext.bits();


	// Encode mode
	encodeMode(decision.mode, globalCoderContext);


	// Intra coding
	if( decision.mode == CodingMode::INTRA)
	{
		{
			// Encode Global
#ifdef ANDROID
			IntraEncodeBow(decision.visualWord, globalCoderContext);
			IntraEncodeKeyPoint(keypoint, globalCoderContext);
#else
			IntraEncodeBowAC(decision.visualWord, globalACCoderContext);
			IntraEncodeKeyPointAC(keypoint, globalACCoderContext);
#endif
			IntraEncodeResidual(decision.residual, globalACCoderContext);


			// Add depth information
			if( !mbCurrentViewRight )
			{
				if( mbMonoDepth )
				{
#ifdef NO_DEPTH_Q
					IntraEncodeDepth(fDepth, globalCoderContext);
#else
					IntraEncodeQuantizedDepth(fDepth, globalCoderContext);
#endif
				}
			}

			const cv::KeyPoint &decKeypoint = fakeCode(keypoint);
			const float qfDepth = fakeCodeDepth(fDepth);
			mCurrentImageBuffer.addFeature(decision.bufferIndex, decKeypoint, descriptor, qfDepth);
		}
	}
	else if( decision.mode == CodingMode::INTER )
	{
		{
			// Encode global
			const int &referenceId = decision.candidate.candidateId;


			// Keypoint
			std::list<ImgBufferEntry>::const_iterator it;
			if( !mbCurrentViewRight )
				it = mLeftImageBuffer.begin();
			else
				it = mRightImageBuffer.begin();

			std::advance(it, decision.candidate.imageId);
			const cv::KeyPoint &refKeypoint = it->mvKeypoints[decision.candidate.keypointId];

#ifdef ANDROID
			stats.interEncStats.bitsReference += encodeReference(referenceId, mInterReferenceImages, globalCoderContext);
#else
			InterEncodeReferenceAC(referenceId, globalACCoderContext);
#endif
			InterEncodeKeypoint(refKeypoint, keypoint, globalACCoderContext, globalCoderContext);
			InterEncodeResidual(decision.residual, globalACCoderContext);

			if( !mbCurrentViewRight )
			{
				// Add depth information, if required - intra-only for now
				if( mbMonoDepth )
				{
#ifdef NO_DEPTH_Q
					IntraEncodeDepth(fDepth, globalCoderContext);
#else
					IntraEncodeQuantizedDepth(fDepth, globalCoderContext);
#endif
				}
			}

			const cv::KeyPoint &decKeypoint = fakeCode(keypoint);
			const float qfDepth = fakeCodeDepth(fDepth);
			mCurrentImageBuffer.addFeature(decision.bufferIndex, decKeypoint, descriptor, qfDepth);
		}
	}
	else if( decision.mode == CodingMode::INTER_SKIP )
	{
		{
			// Encode global
			const int &referenceId = decision.candidate.candidateId;

			// Keypoint
			std::list<ImgBufferEntry>::const_iterator it;
			if( !mbCurrentViewRight )
				it = mLeftImageBuffer.begin();
			else
				it = mRightImageBuffer.begin();

			std::advance(it, decision.candidate.imageId);

			// Get reference feature
			const cv::KeyPoint &refKeypoint = it->mvKeypoints[decision.candidate.keypointId];
			const cv::Mat &refDescriptor = it->mDescriptors.row(decision.candidate.keypointId);
			const float &refDepth = it->mvfDepths[decision.candidate.keypointId];

			InterEncodeReferenceAC(referenceId, globalACCoderContext);


			const cv::KeyPoint &decKeypoint = refKeypoint;
			mCurrentImageBuffer.addFeature(decision.bufferIndex, decKeypoint, refDescriptor, refDepth);
		}
	}
	else if( decision.mode == CodingMode::STEREO_PRED )
	{
		// Encode
		{
			// Encode global
			const int &keypointId = decision.candidate.keypointId;
			StereoEncodeReferenceAC(keypointId, globalACCoderContext);

			const ImgBufferEntry &referenceImage = mLeftImageBuffer.back();
			const cv::KeyPoint &refKeypoint = referenceImage.mvKeypoints[keypointId];
			StereoEncodeKeypointAC(refKeypoint, keypoint, globalACCoderContext, globalCoderContext);
			StereoEncodeResidual(decision.residual, globalACCoderContext);


			const cv::KeyPoint &decKeypoint = fakeCode(keypoint);
			const float qfDepth = -1.0; // No depth for right view :)
			mCurrentImageBuffer.addFeature(decision.bufferIndex, decKeypoint, descriptor, qfDepth);
		}
	}
	const unsigned int bits_end = globalCoderContext.bits() + globalACCoderContext.bits();


	return bits_end - bits_start;
}



void FeatureCoder::decodeFeature(cv::KeyPoint &decKeypoint, cv::Mat &recDescriptor, unsigned int &visualWord,
		DecodeContext &globalDecoderContext, ACDecodeContext &globalACDecoderContext )
{

	CodingMode mode;
	decodeMode(globalDecoderContext, mode);



	if( mode == CodingMode::INTRA)
	{
		// Decode
		cv::Mat residual;
#ifdef ANDROID
		IntraDecodeBow(globalDecoderContext, visualWord);
		IntraDecodeKeyPoint(globalDecoderContext, decKeypoint);
#else
		IntraDecodeBowAC(globalACDecoderContext, visualWord);
		IntraDecodeKeyPointAC(globalACDecoderContext, decKeypoint);
#endif
		IntraDecodeResidual(globalACDecoderContext, residual);
		recDescriptor = IntraReconstructDescriptor(visualWord, residual);

		// Depth data
		if( !mbCurrentViewRight && mbMonoDepth )
		{
			decKeypoint.size = -1.0;
#ifdef NO_DEPTH_Q
			IntraDecodeDepth(globalDecoderContext, decKeypoint.size);
#else
			IntraDecodeQuantizedDepth(globalDecoderContext, decKeypoint.size);
#endif
		}


		decKeypoint.class_id = CodingMode::INTRA;
	}
	else if( mode == CodingMode::INTER )
	{
		// Decode
		cv::Mat residual;
#ifdef ANDROID
		const int recReferenceId = decodeReference(globalDecoderContext, mInterReferenceImages);
#else
		const int recReferenceId = InterDecodeReferenceAC(globalACDecoderContext);
#endif
		// Has to  be N-Sync with  the encoder
		std::list<ImgBufferEntry>::const_iterator it, itEnd;
		if( mbCurrentViewRight )
		{
			it = mRightImageBuffer.begin();
			itEnd = mRightImageBuffer.end();
		}
		else
		{
			it = mLeftImageBuffer.begin();
			itEnd = mLeftImageBuffer.end();
		}

		int keypointId = -1;
		int numKeypoints = 0;
		for(; it != itEnd; it++ )
		{
			if( recReferenceId >= numKeypoints && recReferenceId < numKeypoints + (int) it->mvKeypoints.size())
			{
				keypointId = recReferenceId - numKeypoints;
				break;
			}

			numKeypoints += it->mvKeypoints.size();
		}


		const cv::KeyPoint &recRefKeypoint = it->mvKeypoints[keypointId];
		const cv::Mat &recRefDescriptor =  it->mDescriptors.row(keypointId);


		InterDecodeKeypoint(globalACDecoderContext, globalDecoderContext, recRefKeypoint, decKeypoint);
		InterDecodeResidual(globalACDecoderContext, residual);
		recDescriptor = InterReconstructDescriptor(recRefDescriptor, residual);


		// Depth data
		if( !mbCurrentViewRight && mbMonoDepth )
		{
			decKeypoint.size = -1.0;
#ifdef NO_DEPTH_Q
			IntraDecodeDepth(globalDecoderContext, decKeypoint.size);
#else
			IntraDecodeQuantizedDepth(globalDecoderContext, decKeypoint.size);
#endif
		}

		decKeypoint.class_id = CodingMode::INTER;
		mVoc.transform(recDescriptor, visualWord);
	}
	else if( mode == CodingMode::INTER_SKIP )
	{
		// Decode
		cv::Mat residual;
		const int recReferenceId = InterDecodeReferenceAC(globalACDecoderContext);


		// Has to  be N-Sync with  the encoder
		std::list<ImgBufferEntry>::const_iterator it, itEnd;
		if( mbCurrentViewRight )
		{
			it = mRightImageBuffer.begin();
			itEnd = mRightImageBuffer.end();
		}
		else
		{
			it = mLeftImageBuffer.begin();
			itEnd = mLeftImageBuffer.end();
		}

		int keypointId = -1;
		int numKeypoints = 0;
		for(; it != itEnd; it++ )
		{
			if( recReferenceId >= numKeypoints && recReferenceId < numKeypoints + (int) it->mvKeypoints.size())
			{
				keypointId = recReferenceId - numKeypoints;
				break;
			}

			numKeypoints += it->mvKeypoints.size();
		}


		const cv::KeyPoint &recRefKeypoint = it->mvKeypoints[keypointId];
		cv::Mat recRefDescriptor =  it->mDescriptors.row(keypointId);

		decKeypoint = recRefKeypoint;
		recDescriptor = recRefDescriptor;

		decKeypoint.class_id = CodingMode::INTER_SKIP;
		mVoc.transform(recDescriptor, visualWord);
	}
	else if( mode == CodingMode::STEREO_PRED )
	{
		// Decode
		const ImgBufferEntry &recReferenceImage = mLeftImageBuffer.back();
		const int recKeypointId = StereoDecodeReferenceAC(globalACDecoderContext);

		const cv::KeyPoint &recRefKeypoint = recReferenceImage.mvKeypoints[recKeypointId];
		cv::Mat recRefDescriptor = recReferenceImage.mDescriptors.row(recKeypointId);

		// Keypoint
		StereoDecodeKeypointAC(globalACDecoderContext, globalDecoderContext, recRefKeypoint, decKeypoint);

		// Residual
		cv::Mat recResidual;
		StereoDecodeResidual(globalACDecoderContext, recResidual);
		recDescriptor = InterReconstructDescriptor(recRefDescriptor, recResidual);

		decKeypoint.class_id = CodingMode::STEREO_PRED;
		mVoc.transform(recDescriptor, visualWord);
	}

}



void FeatureCoder::initEncoderModels( ACEncodeContext &accontext )
{
	// Init models

	// INTRA
	ac_model_init (&accontext.acm_bow, mVoc.size(), &mFreqIntraBow[0], 0);
	ac_model_init (&accontext.acm_intra_desc, 2, &mFreqIntraRes[0], 0);
	ac_model_init (&accontext.acm_intra_angle, mAngleBins, &mFreqIntraAngle[0], 0);
	ac_model_init (&accontext.acm_intra_octave, mLevels, &mFreqIntraOctave[0], 0);

	accontext.v_acm_intra_kpt_x.resize(mLevels);
	accontext.v_acm_intra_kpt_y.resize(mLevels);

	for( int octave = 0; octave < mLevels; octave++ )
	{
		ac_model_init (&accontext.v_acm_intra_kpt_x[octave], mFreqIntraPosX[octave].cols, (int *) mFreqIntraPosX[octave].data, 0);
		ac_model_init (&accontext.v_acm_intra_kpt_y[octave], mFreqIntraPosY[octave].cols, (int *) mFreqIntraPosY[octave].data, 0);
	}


	// INTER
	ac_model_init (&accontext.acm_inter_desc, 2, &mFreqInterRes[0], 0);
	ac_model_init (&accontext.acm_inter_angle, mFreqInterAngleDiff.size(), (int *) &mFreqInterAngleDiff[0], 0);
	ac_model_init (&accontext.acm_inter_octave, mFreqInterOctaveDiff.size(), (int *) &mFreqInterOctaveDiff[0], 0);

	accontext.v_acm_inter_kpt.resize(mLevels);
	for( int octave = 0; octave < mLevels; octave++ )
	{
		const int inter_range = mFreqInterKeyPoint[octave].rows*mFreqInterKeyPoint[octave].cols;
		ac_model_init (&accontext.v_acm_inter_kpt[octave], inter_range, (int *) mFreqInterKeyPoint[octave].data, 0);
	}


	// STEREO
	// This is faster than re-initialization
	const int numStereoCandidates = MAX_NUM_FEATURES;
	mFreqStereoCandidate.resize(numStereoCandidates);
	std::fill(mFreqStereoCandidate.begin(), mFreqStereoCandidate.end(), 1);
	ac_model_init (&accontext.acm_stereo_candidate, numStereoCandidates, (int *) &mFreqStereoCandidate[0], 0);

	ac_model_init (&accontext.acm_stereo_desc, 2, &mFreqStereoRes[0], 0);

	accontext.v_acm_stereo_kpt_x.resize(mLevels);
	accontext.v_acm_stereo_kpt_y.resize(mLevels);
	for( int octave = 0; octave < mLevels; octave++ )
	{
		ac_model_init (&accontext.v_acm_stereo_kpt_x[octave], mFreqStereoPosX[octave].cols, (int *) mFreqStereoPosX[octave].data, 0);
		ac_model_init (&accontext.v_acm_stereo_kpt_y[octave], mFreqStereoPosY[octave].cols, (int *) mFreqStereoPosY[octave].data, 0);
	}

	const int numInterCandidates = mBufferSize*MAX_NUM_FEATURES;
	mFreqInterCandidate.resize(numInterCandidates);
	std::fill(mFreqInterCandidate.begin(), mFreqInterCandidate.end(), 1);
	ac_model_init (&accontext.acm_inter_candidate, numInterCandidates, (int *) &mFreqInterCandidate[0], 0);
}



void FeatureCoder::initDecoderModels( ACDecodeContext &accontext )
{
	// INTRA
	ac_model_init (&accontext.acm_bow, mVoc.size(), &mFreqIntraBow[0], 0);
	ac_model_init (&accontext.acm_intra_desc, 2, &mFreqIntraRes[0], 0);
	ac_model_init (&accontext.acm_intra_angle, mAngleBins, &mFreqIntraAngle[0], 0);
	ac_model_init (&accontext.acm_intra_octave, mLevels, &mFreqIntraOctave[0], 0);

	accontext.v_acm_intra_kpt_x.resize(mLevels);
	accontext.v_acm_intra_kpt_y.resize(mLevels);

	for( int octave = 0; octave < mLevels; octave++ )
	{
		ac_model_init (&accontext.v_acm_intra_kpt_x[octave], mFreqIntraPosX[octave].cols, (int *) mFreqIntraPosX[octave].data, 0);
		ac_model_init (&accontext.v_acm_intra_kpt_y[octave], mFreqIntraPosY[octave].cols, (int *) mFreqIntraPosY[octave].data, 0);
	}


	// INTER
	ac_model_init (&accontext.acm_inter_desc, 2, &mFreqInterRes[0], 0);
	ac_model_init (&accontext.acm_inter_angle, mFreqInterAngleDiff.size(), (int *) &mFreqInterAngleDiff[0], 0);
	ac_model_init (&accontext.acm_inter_octave, mFreqInterOctaveDiff.size(), (int *) &mFreqInterOctaveDiff[0], 0);

	accontext.v_acm_inter_kpt.resize(mLevels);
	for( int octave = 0; octave < mLevels; octave++ )
	{
		const int inter_range = mFreqInterKeyPoint[octave].rows*mFreqInterKeyPoint[octave].cols;
		ac_model_init (&accontext.v_acm_inter_kpt[octave], inter_range, (int *) mFreqInterKeyPoint[octave].data, 0);
	}


	// STEREO
	// Faster than adaptive initialization
	const int numStereoCandidates = MAX_NUM_FEATURES;
	mFreqStereoCandidate.resize(numStereoCandidates);
	std::fill(mFreqStereoCandidate.begin(), mFreqStereoCandidate.end(), 1);
	ac_model_init (&accontext.acm_stereo_candidate, numStereoCandidates, (int *) &mFreqStereoCandidate[0], 0);

	ac_model_init (&accontext.acm_stereo_desc, 2, &mFreqStereoRes[0], 0);


	accontext.v_acm_stereo_kpt_x.resize(mLevels);
	accontext.v_acm_stereo_kpt_y.resize(mLevels);
	for( int octave = 0; octave < mLevels; octave++ )
	{
		ac_model_init (&accontext.v_acm_stereo_kpt_x[octave], mFreqStereoPosX[octave].cols, (int *) mFreqStereoPosX[octave].data, 0);
		ac_model_init (&accontext.v_acm_stereo_kpt_y[octave], mFreqStereoPosY[octave].cols, (int *) mFreqStereoPosY[octave].data, 0);
	}

	const int numInterCandidates = mBufferSize*MAX_NUM_FEATURES;
	mFreqInterCandidate.resize(numInterCandidates);
	std::fill(mFreqInterCandidate.begin(), mFreqInterCandidate.end(), 1);
	ac_model_init (&accontext.acm_inter_candidate, numInterCandidates, (int *) &mFreqInterCandidate[0], 0);
}



float FeatureCoder::intraCosts( const cv::KeyPoint &currentKpt, const cv::Mat &descriptor, unsigned int &visualWord, cv::Mat &intraResidualMat)
{
	mVoc.transform(descriptor, visualWord);
	const cv::Mat &visualWordDesc = mVoc.getWord(visualWord);

	cv::bitwise_xor(visualWordDesc, descriptor, intraResidualMat);
	const int d = cv::norm(intraResidualMat, cv::NORM_HAMMING);
	float R_intra_res = mLutRIntra[d];

	const int &octave = currentKpt.octave;
	const float &nbits_x = mvBitsPyramidWidth[octave];
	const float &nbits_y = mvBitsPyramidHeight[octave];
	const float R_intra_kpt = mnBitsAngle + mnBitsOctave + nbits_x + nbits_y;

	return mnBitsBow + R_intra_res + R_intra_kpt;
}



float FeatureCoder::interCandidateSelection( const cv::KeyPoint &currentKpt, const cv::Mat &descriptor, Candidate &cand)
{
	// Search best reference keypoint
	std::list<ImgBufferEntry>::iterator it;
	std::list<ImgBufferEntry>::iterator itEnd;
	if (!mbCurrentViewRight)
	{
		it = mLeftImageBuffer.begin();
		itEnd = mLeftImageBuffer.end();
	}
	else
	{
		it = mRightImageBuffer.begin();
		itEnd = mRightImageBuffer.end();
	}

	std::list<ImgBufferEntry>::iterator bestIt;
	float bestR = std::numeric_limits<float>::max();
	int bestIdx = -1;
	int bestImg = -1;
	int bestReference = -1;

	int img = 0;
	const float R_inter_ref = ceil(log2(MAX_NUM_FEATURES));

	unsigned int candidatesCount = 0;
	for( ; it != itEnd; it++ )
	{
		const std::vector<unsigned int> &vIndices = it->GetFeaturesInArea(currentKpt.pt.x, currentKpt.pt.y, mModel.mSearchRange, currentKpt.octave-1, currentKpt.octave+1);
		for( size_t p = 0; p < vIndices.size(); p++ )
		{
			cv::Mat residual;
			cv::bitwise_xor(it->mDescriptors.row(vIndices[p]), descriptor, residual);
			const int dist = cv::norm(residual, cv::NORM_HAMMING);



			const cv::KeyPoint &refKeypoint = it->mvKeypoints[vIndices[p]];
			const int &octave = currentKpt.octave;

			int refAngleBin = floor(refKeypoint.angle / mAngleBinSize);
			int curAngleBin = floor(currentKpt.angle / mAngleBinSize);


			int angleDiff = curAngleBin - refAngleBin + mnAngleOffset;
			assert( angleDiff >= 0 && angleDiff< mModel.pAngleDelta_.cols);

			float R_angleDiff = -log2(mModel.pAngleDelta_.at<float>(angleDiff));


			// Octave coding
			int octaveDiff = currentKpt.octave - refKeypoint.octave + mnOctaveOffset;
			float R_inter_octave = -log2(mModel.pOctaveDelta_.at<float>(octaveDiff));

			// When octave same use relative coding
			const int sRefx = round(refKeypoint.pt.x / mScaleFactors[octave]);
			const int sRefy = round(refKeypoint.pt.y / mScaleFactors[octave]);

			const int sCurX = round(currentKpt.pt.x / mScaleFactors[octave]);
			const int sCurY = round(currentKpt.pt.y / mScaleFactors[octave]);

			const int dx = sCurX - sRefx;
			const int dy = sCurY - sRefy;

			const int tdx = dx + (mFreqInterKeyPoint[octave].cols-1)/2;
			const int tdy = dy + (mFreqInterKeyPoint[octave].rows-1)/2;

			int index;
			KeyPointDiffToIndex(tdx, tdy, octave, index);

			const float R_inter_res = mLutRInter[dist];

			const float R_inter_xy = -log2(mPInterKeyPoint[octave].at<float>(index));
			const float R_inter_kpt = R_angleDiff + R_inter_octave + R_inter_xy;

			// Skip mode
			float R_inter = R_inter_kpt + R_inter_res + R_inter_ref;
			if( (dist < 5) && (dx == 0) && (dy == 0) && (curAngleBin == refAngleBin) && (currentKpt.octave == refKeypoint.octave) )
			{
				cand.skipMode = true;
				R_inter = R_inter_ref;
			}


			if( R_inter < bestR )
			{
				bestR = R_inter;
				bestIdx = vIndices[p];
				bestImg = img;
				bestReference = candidatesCount + vIndices[p];
				cand.residual = residual;
			}
		}

		candidatesCount +=  it->mvKeypoints.size();
		img++;
	}

	cand.imageId = bestImg;
	cand.keypointId = bestIdx;
	cand.candidateId = bestReference;
	cand.numCandidates = MAX_NUM_FEATURES;

	if( cand.keypointId == -1 )
		return std::numeric_limits<float>::max();

	return bestR;
}



float FeatureCoder::stereoCosts( const cv::KeyPoint &currentKpt, const cv::Mat &descriptor, int stereoBufferId, Candidate &cand)
{
	// Calculate costs
	const ImgBufferEntry &referenceView = mLeftImageBuffer.back();
	cand.keypointId = stereoBufferId;

	const float R_stereo_ref = ceil(log2(MAX_NUM_FEATURES));

	// Calculate stereo cost
	const int &octave = currentKpt.octave;
	const int nbits_x = mvBitsPyramidWidth[octave];
	const int nbits_y = ceil(log2(5));
	float R_stereo_kpt = nbits_x + nbits_y + mnBitsAngle;


	// Residual cost
	cv::Mat refDescriptor = referenceView.mDescriptors.row(stereoBufferId);

	cv::bitwise_xor(refDescriptor, descriptor, cand.residual);
	float d = cv::norm(cand.residual, cv::NORM_HAMMING);
	float R_stereo_res = -(256-d) * log2(mModel.p0_stereo_) - d * log2(1.0 - mModel.p0_stereo_);


	return R_stereo_kpt + R_stereo_res + R_stereo_ref;
}



int FeatureCoder::matchStereoFeatures(const std::vector<cv::KeyPoint> &kptsLeft, const cv::Mat &descriptorsLeft,
		const std::vector<cv::KeyPoint> &kptsRight, const cv::Mat &descriptorsRight,
		std::vector<int> &maskMatch1, std::vector<int> &maskMatch2, std::vector<float> &vfDepthValues)
{
	ImgBufferEntry leftView(mImWidth, mImHeight, mLevels);
	leftView.addFeatures(kptsLeft, descriptorsLeft, vfDepthValues);
	leftView.AssignFeatures();


	int matches = 0;
	for( size_t i = 0; i < kptsRight.size(); i++ )
	{
		int bestDist = 256;
		int bestIdx = -1;

		const cv::KeyPoint &currentKpt = kptsRight[i];

		// Search match in left frame - same octave only
		const std::vector<unsigned int> &vIndices = leftView.GetStereoFeaturesInLine(currentKpt.pt.y, currentKpt.octave);


		int curAngleBin = floor(currentKpt.angle / mAngleBinSize);
		for( size_t p = 0; p < vIndices.size(); p++ )
		{
			const size_t &iL = vIndices[p];
			const cv::KeyPoint &kpL = leftView.mvKeypoints[iL];
			const float &uL = kpL.pt.x;

			int refAngleBin = floor(kpL.angle / mAngleBinSize);

			if( abs(refAngleBin - curAngleBin) > 4 )
				continue;

			if( uL>=currentKpt.pt.x && uL-currentKpt.pt.x <= mfFocalLength )
			{
				const int dist = cv::norm(descriptorsRight.row(i), leftView.mDescriptors.row(vIndices[p]), cv::NORM_HAMMING);
				if( dist < bestDist )
				{
					bestDist = dist;
					bestIdx = vIndices[p];
				}
			}
		}

		if( bestIdx >= 0)
		{
			// Store correspondences
			maskMatch1[bestIdx] = i;
			maskMatch2[i] = bestIdx;
			matches++;


			// Store depth
			const float &uL = kptsLeft[bestIdx].pt.x;
			const float &uR = kptsRight[i].pt.x;

			float disparity = (uL-uR);
			float maxDisp = 200;
			float maxDepth = 100;
			if(disparity>=0 && disparity<maxDisp)
			{
				const float depth = mfBaseLineFocalLength/disparity;

				if( depth < maxDepth )
					vfDepthValues[bestIdx] = depth;
			}

		}
	}


	return matches;
}






ModeDecision FeatureCoder::modeDecision( const cv::KeyPoint &currentKpt, const cv::Mat &descriptor, int stereoBufferId )
{
	unsigned int visualWord;
	cv::Mat intraResidualMat, stereoResidualMask;

	float R_intra = intraCosts(currentKpt, descriptor, visualWord, intraResidualMat);
	float R_inter = std::numeric_limits<float>::max();
	float R_stereo = std::numeric_limits<float>::max();

	Candidate interCandidate;
	Candidate intraPredCandidate;
	Candidate stereoCandidate;

	if( mbInterPred )
		R_inter = interCandidateSelection(currentKpt, descriptor, interCandidate);

	if( stereoBufferId >= 0 )
		R_stereo = stereoCosts(currentKpt, descriptor, stereoBufferId, stereoCandidate);


	ModeDecision decision;
	if( R_intra <= R_inter  && R_intra <= R_stereo )
	{
		decision.visualWord = visualWord;
		decision.mode = CodingMode::INTRA;
		decision.residual = intraResidualMat;
		decision.rate = R_intra;
	}
	if( R_inter <= R_intra   && R_inter <= R_stereo )
	{
		decision.mode = CodingMode::INTER;
		if( interCandidate.skipMode )
			decision.mode = CodingMode::INTER_SKIP;

		decision.residual = interCandidate.residual;
		decision.candidate = interCandidate;
		decision.rate = R_inter;
	}
	if( R_stereo <= R_intra && R_stereo <= R_inter )
	{
		decision.mode = CodingMode::STEREO_PRED;
		decision.residual = stereoCandidate.residual;
		decision.candidate = stereoCandidate;
		decision.rate = R_stereo;
	}


	return decision;
}



size_t FeatureCoder::encodeMode(const CodingMode &mode,  EncodeContext &ctxt)
{
	int nMode = 0;
	if( mode == CodingMode::INTER )
		nMode = 1;
	if( mode == CodingMode::INTER_SKIP )
		nMode = 2;
	if( mode == CodingMode::STEREO_PRED )
		nMode = 3;

	const int nBitsMode = 2;
	for( int i = 0; i < nBitsMode; i++)
	{
		ctxt.cur_bit = ( nMode >> (nBitsMode - i - 1) ) & 0x0001;
		// update the 8-bits buffer
		ctxt.buffer |= ctxt.cur_bit << ctxt.bit_idx;
		ctxt.bit_idx--;

		// when the buffer is full, append it to the vector; then reset the buffer
		if (ctxt.bit_idx<0){
			ctxt.bit_idx = 7;
			ctxt.bitstream.push_back(ctxt.buffer);
			ctxt.buffer = 0;
		}
	}

	return nBitsMode;
}



size_t FeatureCoder::decodeMode(DecodeContext &ctxt, CodingMode &mode)
{
	int nMode = 0;
	const int nBitsMode = 2;
	for( int i = 0; i < nBitsMode; i++ )
	{
		// reset bit counter
		if(ctxt.bit_idx<0){
			ctxt.bit_idx = 7;
			ctxt.cur_byte = ctxt.bitstream[ctxt.byte_idx];
			ctxt.byte_idx++;
		}
		// read the current bit
		ctxt.cur_bit = (ctxt.cur_byte >> ctxt.bit_idx) & 0x01;
		ctxt.bit_idx--;

		nMode |= (ctxt.cur_bit << (nBitsMode - i - 1) );
	}

	if( nMode == 0)
		mode = CodingMode::INTRA;
	else if (nMode == 1 )
		mode = CodingMode::INTER;
	else if (nMode == 2 )
		mode = CodingMode::INTER_SKIP;
	else if( nMode == 3 )
		mode = CodingMode::STEREO_PRED;

	return 1;
}



size_t FeatureCoder::IntraEncodeDepth(const float &fDepth, EncodeContext &ctxt)
{
	int nDepth = (*((int*) &fDepth));
	const int nBitsDepth = sizeof(float)*8;
	for( int i = 0; i < nBitsDepth; i++)
	{
		ctxt.cur_bit = ( nDepth >> (nBitsDepth - i - 1) ) & 0x0001;
		// update the 8-bits buffer
		ctxt.buffer |= ctxt.cur_bit << ctxt.bit_idx;
		ctxt.bit_idx--;

		// when the buffer is full, append it to the vector; then reset the buffer
		if (ctxt.bit_idx<0){
			ctxt.bit_idx = 7;
			ctxt.bitstream.push_back(ctxt.buffer);
			ctxt.buffer = 0;
		}
	}

	return nBitsDepth;
}



void FeatureCoder::IntraDecodeDepth(DecodeContext &ctxt, float &fDepth)
{
	int *depth = (int *) &fDepth;
	(*depth) = 0;

	const int nBitsDepth = sizeof(float)*8;
	for( int i = 0; i < nBitsDepth; i++ )
	{
		// reset bit counter
		if(ctxt.bit_idx<0){
			ctxt.bit_idx = 7;
			ctxt.cur_byte = ctxt.bitstream[ctxt.byte_idx];
			ctxt.byte_idx++;
		}
		// read the current bit
		ctxt.cur_bit = (ctxt.cur_byte >> ctxt.bit_idx) & 0x01;
		ctxt.bit_idx--;

		(*depth) |= (ctxt.cur_bit << (nBitsDepth - i - 1) );
	}
}



size_t FeatureCoder::IntraEncodeQuantizedDepth(const float &fDepth, EncodeContext &ctxt)
{
	// Signal if depth is comming
	ctxt.cur_bit = 0;
	if( (fDepth > 0) && !isinf(fDepth) )
		ctxt.cur_bit = 1;

	// update the 8-bits buffer
	ctxt.buffer |= ctxt.cur_bit << ctxt.bit_idx;
	ctxt.bit_idx--;

	// when the buffer is full, append it to the vector; then reset the buffer
	if (ctxt.bit_idx<0){
		ctxt.bit_idx = 7;
		ctxt.bitstream.push_back(ctxt.buffer);
		ctxt.buffer = 0;
	}


	if( ctxt.cur_bit == 0 )
		return 1;


	// Log search
	size_t idx = Utils::findNearestNeighbourIndex(fDepth, mvfDepthCodeBook);

	for( int i = 0; i < mnDepthBits; i++)
	{
		ctxt.cur_bit = ( idx >> (mnDepthBits - i - 1) ) & 0x0001;
		// update the 8-bits buffer
		ctxt.buffer |= ctxt.cur_bit << ctxt.bit_idx;
		ctxt.bit_idx--;

		// when the buffer is full, append it to the vector; then reset the buffer
		if (ctxt.bit_idx<0){
			ctxt.bit_idx = 7;
			ctxt.bitstream.push_back(ctxt.buffer);
			ctxt.buffer = 0;
		}
	}

	return mnDepthBits + 1;
}



void FeatureCoder::IntraDecodeQuantizedDepth(DecodeContext &ctxt, float &fDepth)
{
	// reset bit counter
	if(ctxt.bit_idx<0){
		ctxt.bit_idx = 7;
		ctxt.cur_byte = ctxt.bitstream[ctxt.byte_idx];
		ctxt.byte_idx++;
	}
	// read the current bit
	ctxt.cur_bit = (ctxt.cur_byte >> ctxt.bit_idx) & 0x01;
	ctxt.bit_idx--;

	// Check if deph available
	if( ctxt.cur_bit == 0 )
		return;


	size_t idx = 0;
	for( int i = 0; i < mnDepthBits; i++ )
	{
		// reset bit counter
		if(ctxt.bit_idx<0){
			ctxt.bit_idx = 7;
			ctxt.cur_byte = ctxt.bitstream[ctxt.byte_idx];
			ctxt.byte_idx++;
		}
		// read the current bit
		ctxt.cur_bit = (ctxt.cur_byte >> ctxt.bit_idx) & 0x01;
		ctxt.bit_idx--;

		idx |= (ctxt.cur_bit << (mnDepthBits - i - 1) );
	}

	// Lookup depth
	fDepth = mvfDepthCodeBook[idx];
}



size_t FeatureCoder::IntraEncodeBow(unsigned int visualWord,  EncodeContext &bowCtxt)
{
	const int bitsBow = ceil(mnBitsBow);
	for( int i = 0; i < bitsBow; i++)
	{
		bowCtxt.cur_bit = ( visualWord >> (bitsBow - i - 1) ) & 0x0001;		// update the 8-bits buffer
		bowCtxt.buffer |= bowCtxt.cur_bit << bowCtxt.bit_idx;
		bowCtxt.bit_idx--;

		// when the buffer is full, append it to the vector; then reset the buffer
		if (bowCtxt.bit_idx<0){
			bowCtxt.bit_idx = 7;
			bowCtxt.bitstream.push_back(bowCtxt.buffer);
			bowCtxt.buffer = 0;
		}
	}

	return mnBitsBow;
}



void FeatureCoder::IntraDecodeBow(DecodeContext &bowCtxt, unsigned int &visualWord )
{
	const int bitsBow = ceil(mnBitsBow);
	visualWord = 0;
	for( int i = 0; i < bitsBow; i++ )
	{
		// reset bit counter
		if(bowCtxt.bit_idx<0){
			bowCtxt.bit_idx = 7;
			bowCtxt.cur_byte = bowCtxt.bitstream[bowCtxt.byte_idx];
			bowCtxt.byte_idx++;
		}
		// read the current bit
		bowCtxt.cur_bit = (bowCtxt.cur_byte >> bowCtxt.bit_idx) & 0x01;
		bowCtxt.bit_idx--;

		visualWord |= (bowCtxt.cur_bit << (bitsBow - i - 1) );
	}

	assert( visualWord <= mVoc.size() );
}



size_t FeatureCoder::IntraEncodeBowAC(unsigned int visualWord,  ACEncodeContext &accontext)
{
	const size_t bits_start = accontext.bits();
	ac_encode_symbol(&accontext.ace, &accontext.acm_bow, visualWord);
	return accontext.bits() - bits_start;
}



void FeatureCoder::IntraDecodeBowAC(ACDecodeContext &accontext, unsigned int &visualWord )
{
	// Setup decoder for descriptor
	visualWord = ac_decode_symbol(&accontext.acd, &accontext.acm_bow);
}



size_t FeatureCoder::IntraEncodeKeyPoint(const cv::KeyPoint &keypoint, EncodeContext &context)
{
	const int &octave = keypoint.octave;

	int angleBin = floor(keypoint.angle / mAngleBinSize);

	const int bitsAngle = ceil(mnBitsAngle);
	for( int i = 0; i < bitsAngle; i++)
	{
		context.cur_bit = ( angleBin >> (bitsAngle - i - 1) ) & 0x0001;
		// update the 8-bits buffer
		context.buffer |= context.cur_bit << context.bit_idx;
		context.bit_idx--;

		// when the buffer is full, append it to the vector; then reset the buffer
		if (context.bit_idx<0){
			context.bit_idx = 7;
			context.bitstream.push_back(context.buffer);
			context.buffer = 0;
		}
	}

	const int bitsOctave = ceil(mnBitsOctave);
	for( int i = 0; i < bitsOctave; i++)
	{
		context.cur_bit = ( octave >> (bitsOctave - i - 1) ) & 0x0001;
		// update the 8-bits buffer
		context.buffer |= context.cur_bit << context.bit_idx;
		context.bit_idx--;

		// when the buffer is full, append it to the vector; then reset the buffer
		if (context.bit_idx<0){
			context.bit_idx = 7;
			context.bitstream.push_back(context.buffer);
			context.buffer = 0;
		}
	}

	// Resize Keypoints to integer resolution
	const int nbits_x = ceil(mvBitsPyramidWidth[octave]);
	const int nbits_y = ceil(mvBitsPyramidHeight[octave]);

	int qx = round(keypoint.pt.x / mScaleFactors[octave]);
	int qy = round(keypoint.pt.y / mScaleFactors[octave]);

	for( int i = 0; i < nbits_x; i++)
	{
		context.cur_bit = ( qx >> (nbits_x - i - 1) ) & 0x0001;
		// update the 8-bits buffer
		context.buffer |= context.cur_bit << context.bit_idx;
		context.bit_idx--;

		// when the buffer is full, append it to the vector; then reset the buffer
		if (context.bit_idx<0){
			context.bit_idx = 7;
			context.bitstream.push_back(context.buffer);
			context.buffer = 0;
		}
	}


	for( int i = 0; i < nbits_y; i++)
	{
		context.cur_bit = ( qy >> (nbits_y - i - 1) ) & 0x0001;
		// update the 8-bits buffer
		context.buffer |= context.cur_bit << context.bit_idx;
		context.bit_idx--;

		// when the buffer is full, append it to the vector; then reset the buffer
		if (context.bit_idx<0){
			context.bit_idx = 7;
			context.bitstream.push_back(context.buffer);
			context.buffer = 0;
		}
	}


	return bitsAngle + bitsOctave + nbits_x + nbits_y;
}



void FeatureCoder::IntraDecodeKeyPoint(DecodeContext &context, cv::KeyPoint &keypoint)
{
	const int bitsAngle = ceil(mnBitsAngle);
	int qangle = 0, qoctave = 0, qx = 0, qy = 0;
	for( int i = 0; i < bitsAngle; i++ )
	{
		// reset bit counter
		if(context.bit_idx<0){
			context.bit_idx = 7;
			context.cur_byte = context.bitstream[context.byte_idx];
			context.byte_idx++;
		}
		// read the current bit
		context.cur_bit = (context.cur_byte >> context.bit_idx) & 0x01;
		context.bit_idx--;

		qangle |= (context.cur_bit << (bitsAngle - i - 1) );
	}

	const int bitsOctave = ceil(mnBitsOctave);
	for( int i = 0; i < bitsOctave; i++){
		// reset bit counter
		if(context.bit_idx<0){
			context.bit_idx = 7;
			context.cur_byte = context.bitstream[context.byte_idx];
			context.byte_idx++;
		}
		// read the current bit
		context.cur_bit = (context.cur_byte >> context.bit_idx) & 0x01;
		context.bit_idx--;

		qoctave |= (context.cur_bit << (bitsOctave - i - 1) );
	}

	const int nbits_x = ceil(mvBitsPyramidWidth[qoctave]);
	const int nbits_y = ceil(mvBitsPyramidHeight[qoctave]);

	for( int i = 0; i <  nbits_x; i++)
	{
		// reset bit counter
		if(context.bit_idx<0){
			context.bit_idx = 7;
			context.cur_byte = context.bitstream[context.byte_idx];
			context.byte_idx++;
		}
		// read the current bit
		context.cur_bit = (context.cur_byte >> context.bit_idx) & 0x01;
		context.bit_idx--;

		qx |= (context.cur_bit << (nbits_x - i - 1) );
	}


	for( int i = 0; i < nbits_y; i++ )
	{
		// reset bit counter
		if(context.bit_idx<0){
			context.bit_idx = 7;
			context.cur_byte = context.bitstream[context.byte_idx];
			context.byte_idx++;
		}
		// read the current bit
		context.cur_bit = (context.cur_byte >> context.bit_idx) & 0x01;
		context.bit_idx--;

		qy |= (context.cur_bit << (nbits_y- i - 1) );
	}


	keypoint.pt.x = (float) qx * mScaleFactors[qoctave];
	keypoint.pt.y = (float) qy * mScaleFactors[qoctave];
	keypoint.octave = qoctave;
	keypoint.angle = (float) qangle * mAngleBinSize + mAngleBinSize / 2;
	assert(keypoint.angle >= 0 && keypoint.angle < 360.0);
};



size_t FeatureCoder::IntraEncodeKeyPointAC(const cv::KeyPoint &keypoint, ACEncodeContext &accontext)
{
	const size_t bits_start = accontext.bits();

	const int &octave = keypoint.octave;
	const int angleBin = floor(keypoint.angle / mAngleBinSize);
	assert( angleBin >= 0 && angleBin <= mAngleBins);


	ac_encode_symbol(&accontext.ace, &accontext.acm_intra_octave, octave);
	ac_encode_symbol(&accontext.ace, &accontext.acm_intra_angle, angleBin);

	// Resize Keypoints to integer resolution
	int qx = round(keypoint.pt.x / mScaleFactors[octave]);
	int qy = round(keypoint.pt.y / mScaleFactors[octave]);

	ac_encode_symbol(&accontext.ace, &accontext.v_acm_intra_kpt_x[octave], qx);
	ac_encode_symbol(&accontext.ace, &accontext.v_acm_intra_kpt_y[octave], qy);

	return accontext.bits() - bits_start;
}



void FeatureCoder::IntraDecodeKeyPointAC(ACDecodeContext &accontext, cv::KeyPoint &keypoint)
{
	const int octave = ac_decode_symbol(&accontext.acd, &accontext.acm_intra_octave);
	const int angleBin = ac_decode_symbol(&accontext.acd, &accontext.acm_intra_angle);

	const int qx = ac_decode_symbol(&accontext.acd, &accontext.v_acm_intra_kpt_x[octave]);
	const int qy = ac_decode_symbol(&accontext.acd, &accontext.v_acm_intra_kpt_y[octave]);

	keypoint.pt.x = (float) qx * mScaleFactors[octave];
	keypoint.pt.y = (float) qy * mScaleFactors[octave];
	keypoint.octave = octave;
	keypoint.angle = (float) angleBin * mAngleBinSize + mAngleBinSize / 2;
	assert(keypoint.angle >= 0 && keypoint.angle < 360.0);
}



size_t FeatureCoder::IntraEncodeResidual(const cv::Mat &residual, ACEncodeContext &accontext)
{
	const size_t bits_start = accontext.bits();

	// Code residuals
	cv::Mat exp_residuals;
	Utils::bin2mat(residual,exp_residuals);

	for( int d = 0; d < exp_residuals.cols; d++ )
	{
		const int &current_bit = exp_residuals.at<uchar>(d);
		ac_encode_symbol(&accontext.ace, &accontext.acm_intra_desc, current_bit);
	}

	return accontext.bits() - bits_start;
}



void FeatureCoder::IntraDecodeResidual(ACDecodeContext &resCtxt, cv::Mat &residual)
{
	// Setup decoder for descriptor
	cv::Mat exp_residuals(1, mModel.mDims, CV_8U);
	for( size_t d = 0; d < mModel.mDims; d++ )
		exp_residuals.at<uchar>(d) = ac_decode_symbol(&resCtxt.acd, &resCtxt.acm_intra_desc);

	cv::Mat binMat;
	Utils::mat2bin(exp_residuals, residual);
}


cv::Mat FeatureCoder::IntraReconstructDescriptor(const unsigned int &visualWord, cv::Mat &residual)
{
	// Reconstruct the descriptor
	const cv::Mat &visualCluster = mVoc.getWord(visualWord);

	cv::Mat descriptor;
	cv::bitwise_xor(residual, visualCluster, descriptor);
	return descriptor;
}



size_t FeatureCoder::InterEncodeReferenceAC(int reference, ACEncodeContext &accontext)
{
	const size_t bits_start = accontext.bits();
	ac_encode_symbol(&accontext.ace, &accontext.acm_inter_candidate, reference);
	return accontext.bits() - bits_start;
}



int FeatureCoder::InterDecodeReferenceAC(ACDecodeContext &accontext)
{
	// First we need the octave and angle.
	const int reference = ac_decode_symbol(&accontext.acd, &accontext.acm_inter_candidate);
	return reference;
}



size_t FeatureCoder::InterEncodeKeypoint(const cv::KeyPoint &refKeypoint, const cv::KeyPoint &currentKeypoint,
		ACEncodeContext &accontext, EncodeContext &context)
{
	const size_t bits_start = accontext.bits();
	const size_t flbits = 0;


	const int refAngleBin = floor(refKeypoint.angle / mAngleBinSize);
	const int curAngleBin = floor(currentKeypoint.angle / mAngleBinSize);
	assert( refAngleBin >= 0 && refAngleBin < mAngleBins);
	assert( curAngleBin >= 0 && curAngleBin < mAngleBins);

	// Angle coding
	const int diff = curAngleBin - refAngleBin;
	assert( diff > -32 && diff < 32);

	const int angleDiff = diff + mnAngleOffset;
	assert(angleDiff >= 0 && angleDiff < (int) mFreqInterAngleDiff.size() );

	ac_encode_symbol(&accontext.ace, &accontext.acm_inter_angle, angleDiff);


	// Octave coding
	const int octaveDiff = currentKeypoint.octave - refKeypoint.octave + mnOctaveOffset;
	ac_encode_symbol(&accontext.ace, &accontext.acm_inter_octave, octaveDiff);


	// KEYPOINT CODING
	const int &octave = currentKeypoint.octave;
	assert( fabs(refKeypoint.pt.x - currentKeypoint.pt.x) <= mModel.mSearchRange );
	assert( fabs(refKeypoint.pt.y - currentKeypoint.pt.y) <= mModel.mSearchRange );


	const int sRefx = round(refKeypoint.pt.x / mScaleFactors[octave]);
	const int sRefy = round(refKeypoint.pt.y / mScaleFactors[octave]);

	const int sCurX = round(currentKeypoint.pt.x / mScaleFactors[octave]);
	const int sCurY = round(currentKeypoint.pt.y / mScaleFactors[octave]);

	const int dx = sCurX - sRefx;
	const int dy = sCurY - sRefy;

	const int tdx = dx + (mFreqInterKeyPoint[octave].cols-1)/2;
	const int tdy = dy + (mFreqInterKeyPoint[octave].rows-1)/2;

	int index;
	KeyPointDiffToIndex(tdx, tdy, octave, index);


	// Coding xy-value
	ac_encode_symbol(&accontext.ace, &accontext.v_acm_inter_kpt[octave], index);


	const size_t acbits = accontext.bits() - bits_start;
	return acbits + flbits;
}



void FeatureCoder::InterDecodeKeypoint(ACDecodeContext &accontext, DecodeContext &context, const cv::KeyPoint &refKeypoint, cv::KeyPoint &currentKeypoint)
{
	// Angle decoding
	const int refAngleBin = floor(refKeypoint.angle / mAngleBinSize);
	const int angleDiff = ac_decode_symbol(&accontext.acd, &accontext.acm_inter_angle);

	// Octave decoding
	const int octaveDiff =  ac_decode_symbol(&accontext.acd, &accontext.acm_inter_octave);

	const int angleBin = angleDiff - mnAngleOffset + refAngleBin;
	const int octave = octaveDiff - mnOctaveOffset + refKeypoint.octave;


	// Keypoint decoding
	int x = 0, y = 0;


	// Position decoding
	const int index = ac_decode_symbol(&accontext.acd, &accontext.v_acm_inter_kpt[octave]);

	int dx,dy;
	IndexToKeyPointDiff(index, octave, dx, dy);

	dx = dx - (mFreqInterKeyPoint[octave].cols-1)/2;
	dy = dy - (mFreqInterKeyPoint[octave].rows-1)/2;

	const int sRefx = round(refKeypoint.pt.x / mScaleFactors[octave]);
	const int sRefy = round(refKeypoint.pt.y / mScaleFactors[octave]);

	x  = (dx + sRefx);
	y  = (dy + sRefy);

	currentKeypoint.pt.x = x *  mScaleFactors[octave];
	currentKeypoint.pt.y = y *  mScaleFactors[octave];
	currentKeypoint.angle = angleBin * mAngleBinSize + mAngleBinSize / 2;
	currentKeypoint.octave = octave;
	assert(currentKeypoint.angle >= 0 && currentKeypoint.angle < 360.0);
}



size_t FeatureCoder::InterEncodeResidual(const cv::Mat &residual, ACEncodeContext &accontext)
{
	const size_t bits_start = accontext.bits();
	cv::Mat expResidual;
	Utils::bin2mat(residual, expResidual);

	for( int d = 0; d < expResidual.cols; d++ )
	{
		const uchar &current_bit = expResidual.at<uchar>(d);
		ac_encode_symbol(&accontext.ace, &accontext.acm_inter_desc, current_bit);
	}

	return accontext.bits() - bits_start;
}



void FeatureCoder::InterDecodeResidual(ACDecodeContext &accontext, cv::Mat&residual)
{
	// We can get the residual vector
	cv::Mat exp_residuals(1, mModel.mDims, CV_8U);
	for( unsigned int d = 0; d < mModel.mDims; d++ )
		exp_residuals.at<uchar>(d) = ac_decode_symbol(&accontext.acd, &accontext.acm_inter_desc);

	Utils::mat2bin(exp_residuals, residual);
}



size_t FeatureCoder::StereoEncodeReferenceAC(int reference, ACEncodeContext &accontext)
{
	const size_t bits_start = accontext.bits();
	ac_encode_symbol(&accontext.ace, &accontext.acm_stereo_candidate, reference);
	return accontext.bits() - bits_start;
}



int FeatureCoder::StereoDecodeReferenceAC(ACDecodeContext &accontext)
{
	// First we need the octave and angle.
	int reference = ac_decode_symbol(&accontext.acd, &accontext.acm_stereo_candidate);
	return reference;
}



size_t FeatureCoder::StereoEncodeKeypointAC(const cv::KeyPoint &refKeypoint, const cv::KeyPoint &currentKeypoint,
		ACEncodeContext &accontext, EncodeContext &context)
{
	const size_t bits_start = accontext.bits();

	assert( refKeypoint.octave == currentKeypoint.octave);

	const int octave = refKeypoint.octave;
	const int angleBin = floor(currentKeypoint.angle / mAngleBinSize);
	assert( angleBin >= 0 && angleBin <= mAngleBins);
	ac_encode_symbol(&accontext.ace, &accontext.acm_intra_angle, angleBin);


	const int qx = round(currentKeypoint.pt.x / mScaleFactors[octave]);
	ac_encode_symbol(&accontext.ace, &accontext.v_acm_intra_kpt_x[octave], qx);

	const int scaledCurrentY = round(currentKeypoint.pt.y / mScaleFactors[octave]);
	const int scaledReferenceY = round(refKeypoint.pt.y / mScaleFactors[octave]);
	int diff = scaledCurrentY - scaledReferenceY + (mFreqStereoPosY[octave].cols-1) / 2;


	ac_encode_symbol(&accontext.ace, &accontext.v_acm_stereo_kpt_y[octave], diff);



	return accontext.bits() - bits_start;
}



void FeatureCoder::StereoDecodeKeypointAC(ACDecodeContext &accontext, DecodeContext &context, const cv::KeyPoint &refKeypoint, cv::KeyPoint &currentKeypoint)
{
	const int angleBin = ac_decode_symbol(&accontext.acd, &accontext.acm_intra_angle);
	const int octave = refKeypoint.octave;

	const int qx = ac_decode_symbol(&accontext.acd, &accontext.v_acm_intra_kpt_x[octave]);
	const int diff = ac_decode_symbol(&accontext.acd, &accontext.v_acm_stereo_kpt_y[octave]);



	const int scaledLeftY = round(refKeypoint.pt.y / mScaleFactors[octave]);
	int qy = scaledLeftY + diff  - (mFreqStereoPosY[octave].cols-1) / 2;


	currentKeypoint.pt.x = (float) qx * mScaleFactors[octave];
	currentKeypoint.pt.y = (float) qy * mScaleFactors[octave];
	currentKeypoint.octave = octave;
	currentKeypoint.angle = (float) angleBin * mAngleBinSize + mAngleBinSize / 2;
	assert(currentKeypoint.angle >= 0 && currentKeypoint.angle < 360.0);
}



size_t FeatureCoder::StereoEncodeKeypoint(const cv::KeyPoint &refKeypoint, const cv::KeyPoint &currentKeypoint,
		ACEncodeContext &accontext, EncodeContext &context)
{
	const int &octave = refKeypoint.octave;

	const int scaledRightX = round(currentKeypoint.pt.x / mScaleFactors[octave]);
	const int scaledLeftX = round(refKeypoint.pt.x / mScaleFactors[octave]);

	int maxVal = ceil(mfFocalLength / mScaleFactors[octave]);
	if( scaledLeftX < maxVal)
		maxVal = scaledLeftX;

	const int nbits_x = ceil(log2(maxVal));
	const int qx = scaledLeftX - scaledRightX;

	assert( qx >= 0 && qx < (int) mfFocalLength);

	for( int i = 0; i < nbits_x; i++)
	{
		context.cur_bit = ( qx >> (nbits_x - i - 1) ) & 0x0001;
		// update the 8-bits buffer
		context.buffer |= context.cur_bit << context.bit_idx;
		context.bit_idx--;

		// when the buffer is full, append it to the vector; then reset the buffer
		if (context.bit_idx<0){
			context.bit_idx = 7;
			context.bitstream.push_back(context.buffer);
			context.buffer = 0;
		}
	}



	const int scaledCurrentY = round(currentKeypoint.pt.y / mScaleFactors[octave]);
	const int scaledRefY = round(refKeypoint.pt.y / mScaleFactors[octave]);

	assert( abs(scaledCurrentY - scaledRefY) <= 4 );

	const int nbits_y = ceil(log2(4+1));
	const int qy = scaledRefY - scaledCurrentY + 2;
	assert( qy >= 0 && qy <= 4);

	for( int i = 0; i < nbits_y; i++)
	{
		context.cur_bit = ( qy >> (nbits_y - i - 1) ) & 0x0001;
		// update the 8-bits buffer
		context.buffer |= context.cur_bit << context.bit_idx;
		context.bit_idx--;

		// when the buffer is full, append it to the vector; then reset the buffer
		if (context.bit_idx<0){
			context.bit_idx = 7;
			context.bitstream.push_back(context.buffer);
			context.buffer = 0;
		}
	}


	return nbits_x + nbits_y;
}



void FeatureCoder::StereoDecodeKeypoint(ACDecodeContext &accontext, DecodeContext &context,  const cv::KeyPoint &refKeypoint, cv::KeyPoint &currentKeypoint)
{
	const int &octave = refKeypoint.octave;
	const int scaledLeftX = round(refKeypoint.pt.x / mScaleFactors[octave]);
	const int scaledLeftY = round(refKeypoint.pt.y / mScaleFactors[octave]);

	int maxVal = ceil(mfFocalLength / mScaleFactors[octave]);
	if( scaledLeftX < maxVal)
		maxVal = scaledLeftX;

	const int nbits_x = ceil(log2(maxVal));

	int qx = 0;
	for( int i = 0; i <  nbits_x; i++)
	{
		// reset bit counter
		if(context.bit_idx<0){
			context.bit_idx = 7;
			context.cur_byte = context.bitstream[context.byte_idx];
			context.byte_idx++;
		}
		// read the current bit
		context.cur_bit = (context.cur_byte >> context.bit_idx) & 0x01;
		context.bit_idx--;

		qx |= (context.cur_bit << (nbits_x - i - 1) );
	}

	int x = scaledLeftX - qx;


	int y = 0;
	const int nbits_y = ceil(log2(4+1));
	for( int i = 0; i <  nbits_y; i++)
	{
		// reset bit counter
		if(context.bit_idx<0){
			context.bit_idx = 7;
			context.cur_byte = context.bitstream[context.byte_idx];
			context.byte_idx++;
		}
		// read the current bit
		context.cur_bit = (context.cur_byte >> context.bit_idx) & 0x01;
		context.bit_idx--;

		y |= (context.cur_bit << (nbits_y - i - 1) );
	}

	y = - y + scaledLeftY + 2;


	currentKeypoint.pt.x = x *  mScaleFactors[octave];
	currentKeypoint.pt.y = y *  mScaleFactors[octave];
	currentKeypoint.angle = refKeypoint.angle;
	currentKeypoint.octave = refKeypoint.octave;
}



size_t FeatureCoder::StereoEncodeResidual(const cv::Mat &residual, ACEncodeContext &accontext)
{
	const size_t bits_start = accontext.bits();

	cv::Mat expResidual;
	Utils::bin2mat(residual, expResidual);


	for( int d = 0; d < expResidual.cols; d++ )
	{
		const int &current_bit = expResidual.at<uchar>(d);
		ac_encode_symbol(&accontext.ace, &accontext.acm_stereo_desc, current_bit);
	}

	return accontext.bits() - bits_start;
}



void FeatureCoder::StereoDecodeResidual(ACDecodeContext &accontext, cv::Mat&residual)
{
	cv::Mat exp_residuals(1, mModel.mDims, CV_8U);
	for( size_t d = 0; d < mModel.mDims; d++ )
		exp_residuals.at<uchar>(d) = ac_decode_symbol(&accontext.acd, &accontext.acm_stereo_desc);

	Utils::mat2bin(exp_residuals, residual);
}



cv::Mat FeatureCoder::InterReconstructDescriptor(const cv::Mat &referenceDescriptor, const cv::Mat &residual)
{
	cv::Mat descriptor;
	cv::bitwise_xor(referenceDescriptor, residual, descriptor);
	return descriptor;
}

void FeatureCoder::KeyPointDiffToIndex(int dx, int dy, int octave, int &index)
{
	index= dy * mFreqInterKeyPoint[octave].cols + dx;
}

void FeatureCoder::IndexToKeyPointDiff(int index, int octave, int &x, int &y)
{
	x = index % mFreqInterKeyPoint[octave].cols;
	y = index / mFreqInterKeyPoint[octave].cols;
}



size_t FeatureCoder::encodeReference(int reference, int numCandidates, EncodeContext &ctxt)
{
	assert( reference < numCandidates );
	const int nBitsReference = ceil(log2(numCandidates));
	for( int i = 0; i < nBitsReference; i++)
	{
		ctxt.cur_bit = ( reference >> (nBitsReference - i - 1) ) & 0x0001;
		// update the 8-bits buffer
		ctxt.buffer |= ctxt.cur_bit << ctxt.bit_idx;
		ctxt.bit_idx--;

		// when the buffer is full, append it to the vector; then reset the buffer
		if (ctxt.bit_idx<0){
			ctxt.bit_idx= 7;
			ctxt.bitstream.push_back(ctxt.buffer);
			ctxt.buffer = 0;
		}
	}

	return nBitsReference;
}



int FeatureCoder::decodeReference(DecodeContext &ctxt, int numCandidates)
{
	// First we need the octave and angle.
	const int nBitsReference = ceil(log2(numCandidates));
	int referenceId = 0;
	for( int i = 0; i < nBitsReference; i++ )
	{
		// reset bit counter
		if(ctxt.bit_idx<0){
			ctxt.bit_idx = 7;
			ctxt.cur_byte = ctxt.bitstream[ctxt.byte_idx];
			ctxt.byte_idx++;
		}
		// read the current bit
		ctxt.cur_bit = (ctxt.cur_byte >> ctxt.bit_idx) & 0x01;
		ctxt.bit_idx--;

		referenceId |= (ctxt.cur_bit << (nBitsReference - i - 1) );
	}

	return referenceId;
}



float FeatureCoder::fakeCodeDepth(const float &fDepth)
{
	size_t idx = Utils::findNearestNeighbourIndex(fDepth, mvfDepthCodeBook);
	return mvfDepthCodeBook[idx];
}



cv::KeyPoint FeatureCoder::fakeCode(const cv::KeyPoint &keyPoint)
{
	// Angle decoding
	const int angleBin = floor(keyPoint.angle / mAngleBinSize);
	assert(angleBin >= 0 && angleBin < mAngleBins);

	cv::KeyPoint decodedKeyPoint;
	decodedKeyPoint.pt = keyPoint.pt;
	decodedKeyPoint.angle = angleBin * mAngleBinSize + mAngleBinSize/2;
	decodedKeyPoint.octave = keyPoint.octave;
	assert( decodedKeyPoint.angle >= 0 && decodedKeyPoint.angle  < 360.0);
	return decodedKeyPoint;
}

} // END NAMESPACE


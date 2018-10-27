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


#pragma once


#include "utils.h"

namespace LBFC2
{

// Coding information
class CodingStats
{
public:
	CodingStats()
	{
		mVer = 0;
		mDims = 0;
		mSearchRange = 10;
		mAngleBins = 32;
		mOctaves = 8;
		mScaleFactor = 1.2f;
		mRespBinSize = 0;
		mDistBinSize = 0;

		mSearchRangeStereoX = 20;
		mSearchRangeStereoY = 20;

		p0_intra_ = 0.8;
		p0_intra_pred_ = 0.8;
		p0_inter_ = 0.8;
		p0_stereo_ = 0.8;
	}


	CodingStats( int dims )
	{
		mVer = 0;
		mDims = dims;
		mSearchRange = 10;
		mAngleBins = 32;
		mOctaves = 8;
		mScaleFactor = 1.2f;
		mRespBinSize = 0;
		mDistBinSize = 0;

		mSearchRangeStereoX = 20;
		mSearchRangeStereoY = 20;

		p0_intra_ = 0.8;
		p0_intra_pred_ = 0.8;
		p0_inter_ = 0.8;
		p0_stereo_ = 0.8;
	}


	void save(const std::string &path)
	{
		FILE *f = fopen(path.c_str(), "w");

		uint64_t v = fwrite(&mDims, sizeof(unsigned int), 1, f);
		v += fwrite(&mSearchRange, sizeof(int), 1, f);
		v += fwrite(&mSearchRangeStereoX, sizeof(int), 1, f);
		v += fwrite(&mSearchRangeStereoY, sizeof(int), 1, f);
		v += fwrite(&mScaleFactor, sizeof(float), 1, f);
		v += fwrite(&mAngleBins, sizeof(int), 1, f);
		v += fwrite(&mOctaves, sizeof(int), 1, f);

		v += fwrite(&p0_intra_, sizeof(float), 1, f);
		v += fwrite(&p0_intra_pred_, sizeof(float), 1, f);
		v += fwrite(&p0_inter_, sizeof(float), 1, f);
		v += fwrite(&p0_stereo_, sizeof(float), 1, f);


		Utils::writeVectorOfMatrix(pPosPerOctave_, f);
		Utils::writeVectorOfMatrix(pPosPerOctaveStereo_, f);

		Utils::writeMatrix(pAngleDelta_, f);
		Utils::writeMatrix(pOctaveDelta_, f);

		// Feature selection
		Utils::writeMatrix(pSigma_, f);
		Utils::writeMatrix(pCoding_, f);
		Utils::writeMatrix(pResp_, f);
		Utils::writeMatrix(pDist_, f);
		Utils::writeMatrix(pBow_, f);

		v += fwrite(&mRespBinSize, sizeof(float), 1, f);
		v += fwrite(&mDistBinSize, sizeof(float), 1, f);


		v += fwrite(&mVer, sizeof(uint32_t), 1, f);

		Utils::writeMatrix(mDepthCodebook, f);
		Utils::writeMatrix(mDepthPartition, f);



		fclose(f);
	}

	void load(const std::string &path)
	{
		FILE *f = fopen(path.c_str(), "r");

		uint64_t v = fread(&mDims, sizeof(unsigned int), 1, f);
		v += fread(&mSearchRange, sizeof(int), 1, f);
		v += fread(&mSearchRangeStereoX, sizeof(int), 1, f);
		v += fread(&mSearchRangeStereoY, sizeof(int), 1, f);
		v += fread(&mScaleFactor, sizeof(float), 1, f);
		v += fread(&mAngleBins, sizeof(int), 1, f);
		v += fread(&mOctaves, sizeof(int), 1, f);

		v += fread(&p0_intra_, sizeof(float), 1, f);
		v += fread(&p0_intra_pred_, sizeof(float), 1, f);
		v += fread(&p0_inter_, sizeof(float), 1, f);
		v += fread(&p0_stereo_, sizeof(float), 1, f);


		Utils::readVectorOfMatrix(pPosPerOctave_, f);
		Utils::readVectorOfMatrix(pPosPerOctaveStereo_, f);

		Utils::readMatrix(pAngleDelta_, f);
		Utils::readMatrix(pOctaveDelta_, f);

		// Feature selection - filled in by matlab
		Utils::readMatrix(pSigma_, f);
		Utils::readMatrix(pCoding_, f);
		Utils::readMatrix(pResp_, f);
		Utils::readMatrix(pDist_, f);
		Utils::readMatrix(pBow_, f);


		if( pSigma_.empty() || pCoding_.empty() || pResp_.empty() || pBow_.empty())
			std::cerr << "Probabilities empty!" << std::endl;

		pSigma_.convertTo(pSigma_, CV_32F);
		pCoding_.convertTo(pCoding_, CV_32F);
		pResp_.convertTo(pResp_, CV_32F);
		pDist_.convertTo(pDist_, CV_32F);
		pBow_.convertTo(pBow_, CV_32F);

		v += fread(&mRespBinSize, sizeof(float), 1, f);
		v += fread(&mDistBinSize, sizeof(float), 1, f);

		uint32_t tmpVer = 0;
		size_t ver_size = fread(&tmpVer, sizeof(uint32_t), 1, f);


		// New version
		if( ver_size > 0 )
		{
			mVer = tmpVer;

			if( mVer >= 1 )
			{
				// Read depth quantization information :)
				Utils::readMatrix(mDepthCodebook, f);
				Utils::readMatrix(mDepthPartition, f);

				mDepthCodebook.convertTo(mDepthCodebook, CV_32F);
				mDepthPartition.convertTo(mDepthPartition, CV_32F);
			}

		}

		fclose(f);
	}


public:
	uint32_t mVer;

	unsigned int mDims;

	// Config
	int mSearchRange;
	int mSearchRangeStereoX;
	int mSearchRangeStereoY;
	float mScaleFactor;
	int mAngleBins;
	int mOctaves;



	// Intra Coding
	float p0_intra_;

	// Intra Pred Coding
	float p0_intra_pred_;

	// Inter Coding
	float p0_inter_;

	// Stereo Coding
	float p0_stereo_;


	std::vector<cv::Mat> pPosPerOctave_;
	std::vector<cv::Mat> pPosPerOctaveStereo_;

	cv::Mat pAngleDelta_;
	cv::Mat pOctaveDelta_;


	// Feature selection parameter
	cv::Mat pSigma_;
	cv::Mat pCoding_;
	cv::Mat pResp_;
	cv::Mat pDist_;
	cv::Mat pBow_;


	float mRespBinSize;
	float mDistBinSize;

	// Depth quantization
	cv::Mat mDepthCodebook;
	cv::Mat mDepthPartition;
};

} // END NAMESPACE


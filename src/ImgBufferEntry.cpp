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


#include "ImgBufferEntry.h"
#include <chrono>
#include <iostream>

namespace LBFC2
{


ImgBufferEntry::ImgBufferEntry(int width, int height, int nLevels, bool leftImage)

{
	mnImageId = 0;
	mnWidth = width;
	mnHeight = height;
	mnLevels = nLevels;

	mbAllocated = false;
	mbLeft = leftImage;

	mfGridElementWidthInv = ((float) mnCols) / mnWidth;
	mfGridElementHeightInv = ((float) mnRows) / mnHeight;

	const int res = 1200.0f / (mnCols*mnRows);

	mGrid.resize(mnCols);
	for(int i=0; i<mnCols;i++)
	{
		mGrid[i].resize(mnRows);
		for(int j=0; j<mnRows;j++)
			mGrid[i][j].reserve(res);
	}


	mvKeypoints.reserve(1200);
	mvVisualWords.reserve(1200);
}

void ImgBufferEntry::allocateSpace( const int nFeatures)
{
	mvKeypoints.resize(nFeatures);
	mvVisualWords.resize(nFeatures);
	mDescriptors = cv::Mat(nFeatures, 32, CV_8U);
	mvfDepths.resize(nFeatures, -1.0);
	mbAllocated = true;

}
void ImgBufferEntry::addFeature( const int index, const cv::KeyPoint &kp, const cv::Mat &descriptor, const float &fDepth )
{
	assert(mbAllocated);
	mvKeypoints[index] = kp;
	descriptor.copyTo(mDescriptors.row(index));
	mvfDepths[index] = fDepth;
}

void ImgBufferEntry::addFeature( const int index, const cv::KeyPoint &kp, const cv::Mat &descriptor, const unsigned int visualWord, const float &fDepth )
{
	assert(mbAllocated);
	mvKeypoints[index] = kp;
	mvVisualWords[index] = visualWord;
	descriptor.copyTo(mDescriptors.row(index));
	mvfDepths[index] = fDepth;
}


void ImgBufferEntry::addFeatures( const std::vector<cv::KeyPoint> &kpts, const cv::Mat &descriptor, const std::vector<float> &vfDepths)
{
	for( auto &kp : kpts )
	{
		int nGridPosX, nGridPosY;
		if(PosInGrid(kp,nGridPosX,nGridPosY))
			mGrid[nGridPosX][nGridPosY].push_back(mvKeypoints.size());
	}

	mvKeypoints = kpts;
	mDescriptors.push_back(descriptor);
	mvfDepths = vfDepths;
}


bool ImgBufferEntry::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
	// From ORB SLAM
	posX = round(kp.pt.x*mfGridElementWidthInv);
	posY = round(kp.pt.y*mfGridElementHeightInv);

	if( posX < 0 || posY < 0 || posX >= mnCols || posY >= mnRows )
		return false;

	return true;
}


std::vector<unsigned int> ImgBufferEntry::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
	// From ORB SLAM
	std::vector<unsigned int> vIndices;

	const int nMinCellX = std::max(0,(int)floor((x-r)*mfGridElementWidthInv));
	if(nMinCellX>=mnCols)
		return vIndices;

	const int nMaxCellX = std::min((int)mnCols-1,(int)ceil((x+r)*mfGridElementWidthInv));
	if(nMaxCellX<0)
		return vIndices;

	const int nMinCellY = std::max(0,(int)floor((y-r)*mfGridElementHeightInv));
	if(nMinCellY>=mnRows)
		return vIndices;

	const int nMaxCellY = std::min((int)mnRows-1,(int)ceil((y+r)*mfGridElementHeightInv));
	if(nMaxCellY<0)
		return vIndices;

	const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

	for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
	{
		for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
		{
			const std::vector<unsigned int> &vCell = mGrid[ix][iy];
			if(vCell.empty())
				continue;

			// loop through all mvKeypoints in the current grid cell
			for(size_t j=0, jend=vCell.size(); j<jend; j++)
			{
				const cv::KeyPoint &kpUn = mvKeypoints[vCell[j]];
				if(bCheckLevels)
				{
					if(kpUn.octave<minLevel)
						continue;
					if(maxLevel>=0)
						if(kpUn.octave>maxLevel)
							continue;
				}

				const float distx = kpUn.pt.x-x;
				const float disty = kpUn.pt.y-y;

				// save the keypoint when it is within the search radius
				if(fabs(distx)<=r && fabs(disty)<=r)
					vIndices.push_back(vCell[j]);
			}
		}
	}
	return vIndices;
}


void ImgBufferEntry::AssignFeatures()
{
	//Assign mvKeypoints to row table
	mvRowIndices.resize(mnLevels);
	for( int o=0; o < mnLevels; o++)
	{
		mvRowIndices[o].resize(mnHeight);
		for(int i=0; i<mnHeight; i++)
			mvRowIndices[o][i].reserve(200);
	}


	const int Nr = mvKeypoints.size();

	for(int iR=0; iR<Nr; iR++)
	{
		const cv::KeyPoint &kp = mvKeypoints[iR];
		const int &octave = kp.octave;
		const float &kpY = kp.pt.y;
		const float r = 2.0f*pow(1.2f,mvKeypoints[iR].octave);
		const int maxr = floor(kpY+r);
		const int minr = ceil(kpY-r);

		for(int yi=minr;yi<=maxr;yi++)
			mvRowIndices[octave][yi].push_back(iR);
	}


	for(int i=0; i<Nr; i++)
	{
		int nGridPosX, nGridPosY;
		if(PosInGrid(mvKeypoints[i],nGridPosX,nGridPosY))
			mGrid[nGridPosX][nGridPosY].push_back(i);
	}
}


std::vector<unsigned int> ImgBufferEntry::GetStereoFeaturesInLine(const float  &yL, const int &octave) const
{
	return mvRowIndices[octave][yL];
}

} // END NAMESPACE

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


#include <opencv2/opencv.hpp>

namespace LBFC2
{

class ImgBufferEntry
{
public:
	ImgBufferEntry(int width = -1, int height = -1, int nLevels = 8, bool leftImage = true);

	void allocateSpace( const int nFeatures);

	void addFeature( const int index, const cv::KeyPoint &kp, const cv::Mat &descriptor, const float &fDepth);
	void addFeature( const int index, const cv::KeyPoint &kp, const cv::Mat &descriptor, const unsigned int visualWord, const float &fDepth);
	void addFeatures( const std::vector<cv::KeyPoint> &kpts, const cv::Mat &descriptor, const std::vector<float> &vfDepths);


	bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);
	void AssignFeatures();
	std::vector<unsigned int> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const;
	std::vector<unsigned int> GetStereoFeaturesInLine(const float  &yL, const int &octave) const;

public:
	cv::Mat mDescriptors;
	std::vector<cv::KeyPoint> mvKeypoints;
	std::vector<unsigned int> mvVisualWords;
	std::vector<float> mvfDepths;

	int mnWidth;
	int mnHeight;
	int mnLevels;

	bool mbAllocated;
	bool mbLeft;
	long long mnImageId;

	int mnCols = 32;
	int mnRows = 24;
	float mfGridElementWidthInv;
	float mfGridElementHeightInv;
	std::vector<std::vector<std::vector<unsigned int> > > mGrid;
	std::vector<std::vector<std::vector<unsigned int> > > mvRowIndices;


};

} // END NAMESPACE

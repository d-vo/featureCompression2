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


#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

namespace LBFC2
{


class Utils
{
public:
	static bool writeMatrix(const cv::Mat &mat, const std::string &path);
	static bool readMatrix(cv::Mat &mat, const std::string &path);

	static bool writeMatrix(const cv::Mat &mat, FILE *f);
	static bool readMatrix(cv::Mat &mat, FILE *f);

	static bool writeVectorOfMatrix(const std::vector<cv::Mat> &mat, FILE *f);
	static bool readVectorOfMatrix(std::vector<cv::Mat> &mat, FILE *f);


	static void bin2mat( cv::InputArray _src, cv::OutputArray _dst, int type = CV_8U, bool norm = false  );
	static void mat2bin( cv::InputArray _src, cv::OutputArray _dst );


	template<typename T>
	static size_t findNearestNeighbourIndex(const T &value, const std::vector<T> &x)
	{
		typename std::vector<T>::const_iterator first = x.begin();
		typename std::vector<T>::const_iterator last = x.end();
		typename std::vector<T>::const_iterator before = std::lower_bound(first, last, value);
		typename std::vector<T>::const_iterator it;

		if (before == first)
			it = first;
		else if (before == last)
			it = --last;
		else
		{
			typename std::vector<T>::const_iterator after = before;
			--before;
			it = (*after - value) < (value - *before) ? after : before;
		}

		return std::distance(first, it);
	}

};

} // END NAMESPACE


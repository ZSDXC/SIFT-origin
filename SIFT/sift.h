#pragma once
#include "opencv2/features2d/features2d.hpp"
#include<opencv2\opencv.hpp>
#include<iostream>
#include<fstream>
using namespace cv;
namespace cv
{

	class CV_EXPORTS_W SIFT : public Feature2D
	{
	public:
		CV_PROP_RW int nfeatures;
		CV_PROP_RW int nOctaveLayers;
		CV_PROP_RW double contrastThreshold;
		CV_PROP_RW double edgeThreshold;
		CV_PROP_RW double sigma;
		explicit SIFT(int nfeatures = 0, int nOctaveLayers = 3,
			double contrastThreshold = 0.04, double edgeThreshold = 10,
			double sigma = 1.6);

		//! returns the descriptor size in floats (128)
		//����������ά��
		int descriptorSize() const;

		//! returns the descriptor type
		//��������������
		int descriptorType() const;

		//! finds the keypoints using SIFT algorithm
		//���ز�����()����SIFT�㷨�ҵ��ؼ���
		void operator()(InputArray img, InputArray mask,
			vector<KeyPoint>& keypoints) const;


		//! finds the keypoints and computes descriptors for them using SIFT algorithm.
		//! Optionally it can compute descriptors for the user-provided keypoints
		//���ز�����()����SIFT�㷨�ҹؼ��㲢�������������������һ���������Լ����û��Լ��ṩ���������������
		// mask ��Optional input mask that marks the regions where we should detect features.
		void operator()(InputArray img, InputArray mask,
			vector<KeyPoint>& keypoints,
			OutputArray descriptors,
			bool useProvidedKeypoints = false) const;

		AlgorithmInfo* info() const;

		static Mat createInitialImage(const Mat& img, bool doubleImageSize, float sigma);
		void buildGaussianPyramid(const Mat& base, vector<Mat>& pyr, int nOctaves) const;
		void buildDoGPyramid(const	vector<Mat>& pyr, vector<Mat>& dogpyr) const;
		void findScaleSpaceExtrema(const vector<Mat>& gauss_pyr, const vector<Mat>& dog_pyr,
			vector<KeyPoint>& keypoints) const;
		static void calcSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl,
			int d, int n, float* dst);
		static void calcDescriptors(const vector<Mat>& gpyr, const vector<KeyPoint>& keypoints,
			Mat& descriptors, int nOctaveLayers);

	protected:
		void detectImpl(const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask = Mat()) const;
		void computeImpl(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) const;


		// CV_PROP_RW bool doubleImageSize=false;
	};
}

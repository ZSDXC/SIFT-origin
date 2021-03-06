#include "stdafx.h"
#include <vector>
#include <stdarg.h>
#include"sift.h"
#include<opencv2\opencv.hpp>
#include<opencv2\features2d\features2d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include<iostream>
#include<fstream>
#include"sift.h"

using namespace cv;
using namespace std;
// default number of sampled intervals per octave
static const int SIFT_INTVLS = 3;

// default sigma for initial gaussian smoothing
static const float SIFT_SIGMA = 1.6f;

// default threshold on keypoint contrast |D(x)|
static const float SIFT_CONTR_THR = 0.04f;

// default threshold on keypoint ratio of principle curvatures
static const float SIFT_CURV_THR = 10.f;

// double image size before pyramid construction?
static const bool SIFT_IMG_DBL = true;

// default width of descriptor histogram array
static const int SIFT_DESCR_WIDTH = 4;

// default number of bins per histogram in descriptor array
static const int SIFT_DESCR_HIST_BINS = 8;

// assumed gaussian blur for input image
static const float SIFT_INIT_SIGMA = 0.5f;

// width of border in which to ignore keypoints
static const int SIFT_IMG_BORDER = 5;

// maximum steps of keypoint interpolation before failure
static const int SIFT_MAX_INTERP_STEPS = 5;

// default number of bins in histogram for orientation assignment
static const int SIFT_ORI_HIST_BINS = 36;

// determines gaussian sigma for orientation assignment
static const float SIFT_ORI_SIG_FCTR = 1.5f;

// determines the radius of the region used in orientation assignment
static const float SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR;

// orientation magnitude relative to max that results in new feature
static const float SIFT_ORI_PEAK_RATIO = 0.8f;

// determines the size of a single descriptor orientation histogram
static const float SIFT_DESCR_SCL_FCTR = 3.f;

// threshold on magnitude of elements of descriptor vector
static const float SIFT_DESCR_MAG_THR = 0.2f;

// factor used to convert floating-point descriptor to unsigned char
static const float SIFT_INT_DESCR_FCTR = 512.f;

static const int SIFT_FIXPT_SCALE = 48;



//std::ofstream fout("sigma.txt");   //保存尺度
//成功
Mat SIFT::createInitialImage(const Mat& img, bool doubleImageSize, float sigma)
{
	Mat gray, gray_fpt;
	if (img.channels() == 3 || img.channels() == 4)
		cvtColor(img, gray, COLOR_BGR2GRAY); //原始图像转灰度
	else
		img.copyTo(gray);
	//缩放并转换到另外一种数据类型,深度转换为CV_16S避免外溢。（48,0）为缩放参数
	//灰度值拉伸了48倍,CV_16S避免外溢
	gray.convertTo(gray_fpt, CV_16S, SIFT_FIXPT_SCALE, 0); //SIFT_FIXPT_SCALE=48


	float sig_diff;

	//默认传进来的doubleImageSIze不是flase吗？这里应该是if(!doubleImageSize)啊？？
	if (doubleImageSize)
	{
		//sigma=1.6,SIFT_INIT_SIGMA=0.5
		sig_diff = sqrtf(std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01f));
		Mat dbl;
		resize(gray_fpt, dbl, Size(gray.cols * 2, gray.rows * 2), 0, 0, INTER_LINEAR); //长宽乘2
		GaussianBlur(dbl, dbl, Size(), sig_diff, sig_diff);
		return dbl;
	}
	else
	{
		sig_diff = sqrtf(std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f));
		GaussianBlur(gray_fpt, gray_fpt, Size(), sig_diff, sig_diff);
		return gray_fpt;
	}
}
//构建高斯金字塔
//成功
void SIFT::buildGaussianPyramid(const Mat& base, vector<Mat>& pyr, int nOctaves) const
{
	vector<double> sig(nOctaveLayers + 3);
	pyr.resize(nOctaves*(nOctaveLayers + 3)); //pyr保存所有组所有层

											   // precompute Gaussian sigmas using the following formula:
											   //  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
											   //计算第0组每层的尺度因子sig[i]，第0组第0层已经模糊过，所有只有5层需要模糊
	sig[0] = sigma;						//第0层尺度为sigma
								   //fout<<"每层的尺度为：\n";
								   //fout<<sig[0]<<'\n';
	double k = pow(2., 1. / nOctaveLayers);
	for (int i = 1; i < nOctaveLayers + 3; i++)
	{
		double sig_prev = pow(k, (double)(i - 1))*sigma;
		double sig_total = sig_prev * k;
		sig[i] = std::sqrt(sig_total*sig_total - sig_prev * sig_prev);
		// fout<<sig[i]<<'\n';
	}


	//512大小的图像，nOctaves=7;
	for (int o = 0; o < nOctaves; o++)
	{
		for (int i = 0; i < nOctaveLayers + 3; i++)
		{
			Mat& dst = pyr[o*(nOctaveLayers + 3) + i];

			//第0组第0层为base层，即原始图像
			if (o == 0 && i == 0)
				dst = base;

			// base of new octave is halved image from end of previous octave
			//高斯金字塔的新组(new octave)的第0幅为上一组的第nOctaveLayers幅下采样得到，采样步长为2
			else if (i == 0)
			{
				const Mat& src = pyr[(o - 1)*(nOctaveLayers + 3) + nOctaveLayers];
				resize(src, dst, Size(src.cols / 2, src.rows / 2),
					0, 0, INTER_NEAREST);
			}

			// 每一组的第i幅图像是由该组第i-1幅图像用sig[i]高斯模糊得到，相当于使用了新的尺度。
			else
			{
				const Mat& src = pyr[o*(nOctaveLayers + 3) + i - 1];
				GaussianBlur(src, dst, Size(), sig[i], sig[i]);
			}
		}
	}
}
//构建DOG
//成功
void SIFT::buildDoGPyramid(const vector<Mat>& gpyr, vector<Mat>& dogpyr) const
{
	int nOctaves = (int)gpyr.size() / (nOctaveLayers + 3); //nOctaves表示组的个数
	dogpyr.resize(nOctaves*(nOctaveLayers + 2)); //保存所有组的Dog图像


													//每组相邻两幅图像相减，获取Dog图像
	for (int o = 0; o < nOctaves; o++)
	{
		for (int i = 0; i < nOctaveLayers + 2; i++)
		{
			const Mat& src1 = gpyr[o*(nOctaveLayers + 3) + i];
			const Mat& src2 = gpyr[o*(nOctaveLayers + 3) + i + 1];
			Mat& dst = dogpyr[o*(nOctaveLayers + 2) + i];
			subtract(src2, src1, dst, noArray(), CV_16S); //两幅图像相减
		}
	}
}


//计算某一个特征点的周围区域梯度方向直方图
// Computes a gradient orientation histogram at a specified pixel
static float calcOrientationHist(const Mat& img, Point pt, int radius,
	float sigma, float* hist, int n)
{
	int i, j, k, len = (radius * 2 + 1)*(radius * 2 + 1);

	float expf_scale = -1.f / (2.f * sigma * sigma);
	AutoBuffer<float> buf(len * 4 + n + 4);
	float *X = buf, *Y = X + len, *Mag = X, *Ori = Y + len, *W = Ori + len;
	float* temphist = W + len + 2;

	for (i = 0; i < n; i++)
		temphist[i] = 0.f;

	// 图像梯度直方图统计的像素范围
	for (i = -radius, k = 0; i <= radius; i++)
	{
		int y = pt.y + i;
		if (y <= 0 || y >= img.rows - 1)
			continue;
		for (j = -radius; j <= radius; j++)
		{
			int x = pt.x + j;
			if (x <= 0 || x >= img.cols - 1)
				continue;

			float dx = (float)(img.at<short>(y, x + 1) - img.at<short>(y, x - 1));
			float dy = (float)(img.at<short>(y - 1, x) - img.at<short>(y + 1, x));

			X[k] = dx; Y[k] = dy; W[k] = (i*i + j * j)*expf_scale;
			k++;
		}
	}

	len = k;

	// compute gradient values, orientations and the weights over the pixel neighborhood
	exp(W, W, len);
	fastAtan2(Y, X, Ori, len, true);
	magnitude(X, Y, Mag, len);

	// 计算直方图的每个bin
	for (k = 0; k < len; k++)
	{
		int bin = cvRound((n / 360.f)*Ori[k]);
		if (bin >= n)
			bin -= n;
		if (bin < 0)
			bin += n;
		temphist[bin] += W[k] * Mag[k];
	}

	// smooth the histogram
	// 高斯平滑
	temphist[-1] = temphist[n - 1];
	temphist[-2] = temphist[n - 2];
	temphist[n] = temphist[0];
	temphist[n + 1] = temphist[1];
	for (i = 0; i < n; i++)
	{
		hist[i] = (temphist[i - 2] + temphist[i + 2])*(1.f / 16.f) +
			(temphist[i - 1] + temphist[i + 1])*(4.f / 16.f) +
			temphist[i] * (6.f / 16.f);
	}

	// 得到主方向
	float maxval = hist[0];
	for (i = 1; i < n; i++)
		maxval = std::max(maxval, hist[i]);

	return maxval;
}

//  dog_pyr:dog金字塔;
//  kpt:关键点;
//  octv:组序号
//  layer: dog层序号
//  r: 行号; c:列号
//  nOctaveLayers:dog中要用到的层数，为3
//  contrastThreshold:对比度阈值=0.04
//   edgeThreshold:边界阈值=10
//  sigma: 尺度因子
static bool adjustLocalExtrema(const vector<Mat>& dog_pyr, KeyPoint& kpt, int octv,
	int& layer, int& r, int& c, int nOctaveLayers,
	float contrastThreshold, float edgeThreshold, float sigma)
{
	const float img_scale = 1.f / (255 * SIFT_FIXPT_SCALE);
	const float deriv_scale = img_scale * 0.5f;
	const float second_deriv_scale = img_scale;
	const float cross_deriv_scale = img_scale * 0.25f;

	float xi = 0, xr = 0, xc = 0, contr = 0;
	int i = 0;

	for (; i < SIFT_MAX_INTERP_STEPS; i++)
	{
		int idx = octv * (nOctaveLayers + 2) + layer;
		const Mat& img = dog_pyr[idx];
		const Mat& prev = dog_pyr[idx - 1];
		const Mat& next = dog_pyr[idx + 1];

		/*Vec3f dD((img.at<sift_wt>(r, c + 1) - img.at<sift_wt>(r, c - 1))*deriv_scale,
			(img.at<sift_wt>(r + 1, c) - img.at<sift_wt>(r - 1, c))*deriv_scale,
			(next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c))*deriv_scale);

		float v2 = (float)img.at<sift_wt>(r, c) * 2;
		float dxx = (img.at<sift_wt>(r, c + 1) + img.at<sift_wt>(r, c - 1) - v2)*second_deriv_scale;
		float dyy = (img.at<sift_wt>(r + 1, c) + img.at<sift_wt>(r - 1, c) - v2)*second_deriv_scale;
		float dss = (next.at<sift_wt>(r, c) + prev.at<sift_wt>(r, c) - v2)*second_deriv_scale;
		float dxy = (img.at<sift_wt>(r + 1, c + 1) - img.at<sift_wt>(r + 1, c - 1) -
			img.at<sift_wt>(r - 1, c + 1) + img.at<sift_wt>(r - 1, c - 1))*cross_deriv_scale;
		float dxs = (next.at<sift_wt>(r, c + 1) - next.at<sift_wt>(r, c - 1) -
			prev.at<sift_wt>(r, c + 1) + prev.at<sift_wt>(r, c - 1))*cross_deriv_scale;
		float dys = (next.at<sift_wt>(r + 1, c) - next.at<sift_wt>(r - 1, c) -
			prev.at<sift_wt>(r + 1, c) + prev.at<sift_wt>(r - 1, c))*cross_deriv_scale;*/

		Vec3f dD((img.at<short>(r, c + 1) - img.at<short>(r, c - 1))*deriv_scale,
			(img.at<short>(r + 1, c) - img.at<short>(r - 1, c))*deriv_scale,
			(next.at<short>(r, c) - prev.at<short>(r, c))*deriv_scale);

		float v2 = (float)img.at<short>(r, c) * 2;
		float dxx = (img.at<short>(r, c + 1) +
			img.at<short>(r, c - 1) - v2)*second_deriv_scale;
		float dyy = (img.at<short>(r + 1, c) +
			img.at<short>(r - 1, c) - v2)*second_deriv_scale;
		float dss = (next.at<short>(r, c) +
			prev.at<short>(r, c) - v2)*second_deriv_scale;
		float dxy = (img.at<short>(r + 1, c + 1) -
			img.at<short>(r + 1, c - 1) - img.at<short>(r - 1, c + 1) +
			img.at<short>(r - 1, c - 1))*cross_deriv_scale;
		float dxs = (next.at<short>(r, c + 1) -
			next.at<short>(r, c - 1) - prev.at<short>(r, c + 1) +
			prev.at<short>(r, c - 1))*cross_deriv_scale;
		float dys = (next.at<short>(r + 1, c) -
			next.at<short>(r - 1, c) - prev.at<short>(r + 1, c) +
			prev.at<short>(r - 1, c))*cross_deriv_scale;



		Matx33f H(dxx, dxy, dxs,
			dxy, dyy, dys,
			dxs, dys, dss);

		Vec3f X = H.solve(dD, DECOMP_LU);

		//H.solve(dD)用于求解线性方程组，该方程组是  dD = H * X 。
			//前面提到极值点的方程是 ： G0 = H0 * (-△) ，这里G0就是dD, H0就是H，(-△)就是X。因此为求得△，只需对方程解X求反。如下面的xi、xr和xc。

			xi = -X[2];
		xr = -X[1];
		xc = -X[0];

		if (std::abs(xi) < 0.5f && std::abs(xr) < 0.5f && std::abs(xc) < 0.5f)
			break;

		if (std::abs(xi) > (float)(INT_MAX / 3) ||
			std::abs(xr) > (float)(INT_MAX / 3) ||
			std::abs(xc) > (float)(INT_MAX / 3))
			return false;

		c += cvRound(xc);
		r += cvRound(xr);
		layer += cvRound(xi);


		//如果拟合的极值点与当前的整数点偏差大于0.5，则说明真正的极值点偏离当前整数点很多，需要换一个更好的整数点，然后重新拟合。知道偏差小于0.5或迭代超过一定次数。



			if (layer < 1 || layer > nOctaveLayers ||
				c < SIFT_IMG_BORDER || c >= img.cols - SIFT_IMG_BORDER ||
				r < SIFT_IMG_BORDER || r >= img.rows - SIFT_IMG_BORDER)
				return false;
	}

	// ensure convergence of interpolation
	if (i >= SIFT_MAX_INTERP_STEPS)
		return false;

	{
		int idx = octv * (nOctaveLayers + 2) + layer;
		const Mat& img = dog_pyr[idx];
		const Mat& prev = dog_pyr[idx - 1];
		const Mat& next = dog_pyr[idx + 1];
		Matx31f dD((img.at<short>(r, c + 1) - img.at<short>(r, c - 1))*deriv_scale,
			(img.at<short>(r + 1, c) - img.at<short>(r - 1, c))*deriv_scale,
			(next.at<short>(r, c) - prev.at<short>(r, c))*deriv_scale);
		float t = dD.dot(Matx31f(xc, xr, xi));

		contr = img.at<short>(r, c)*img_scale + t * 0.5f;
		if (std::abs(contr) * nOctaveLayers < contrastThreshold)
			return false;

		/* principal curvatures are computed using the trace and det of Hessian */
		//利用Hessian矩阵的迹和行列式计算主曲率的比值
		float v2 = img.at<short>(r, c)*2.f;
		float dxx = (img.at<short>(r, c + 1) +
			img.at<short>(r, c - 1) - v2)*second_deriv_scale;
		float dyy = (img.at<short>(r + 1, c) +
			img.at<short>(r - 1, c) - v2)*second_deriv_scale;
		float dxy = (img.at<short>(r + 1, c + 1) -
			img.at<short>(r + 1, c - 1) - img.at<short>(r - 1, c + 1) +
			img.at<short>(r - 1, c - 1)) * cross_deriv_scale;
		float tr = dxx + dyy;
		float det = dxx * dyy - dxy * dxy;

		//这里edgeThreshold可以在调用SIFT()时输入；
		//其实代码中定义了 static const float SIFT_CURV_THR = 10.f 可以直接使用
		if (det <= 0 || tr * tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det)
			return false;
	}

	kpt.pt.x = (c + xc) * (1 << octv);
	kpt.pt.y = (r + xr) * (1 << octv);
	kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5) * 255) << 16);
	kpt.size = sigma * powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv) * 2;

	return true;
}


void SIFT::findScaleSpaceExtrema(const vector<Mat>& gauss_pyr, const vector<Mat>& dog_pyr,
	vector<KeyPoint>& keypoints) const
{
	int nOctaves = (int)gauss_pyr.size() / (nOctaveLayers + 3); //组的个数

																   // The contrast threshold used to filter out weak features in semi-uniform  
																   // (low-contrast) regions. The larger the threshold, the less features are produced by the detector.  
																   //低 对比度的阈值， contrastThreshold默认为0.04  
	int threshold = cvFloor(0.5 * contrastThreshold / nOctaveLayers * 255 * SIFT_FIXPT_SCALE);
	const int n = SIFT_ORI_HIST_BINS; //直方图bin的个数=36，每个10度
	float hist[n];
	KeyPoint kpt;

	keypoints.clear();

	for (int o = 0; o < nOctaves; o++)
		for (int i = 1; i <= nOctaveLayers; i++) // nOctaveLayers表示每层Dog图像个数-2，即最终用到的层数
		{
			int idx = o * (nOctaveLayers + 2) + i;
			const Mat& img = dog_pyr[idx]; //获取该层Dog图像，序号从1开始。第0层和最后一层不用
			const Mat& prev = dog_pyr[idx - 1]; //上一幅Dog图像
			const Mat& next = dog_pyr[idx + 1]; //下一幅Dog图像
			int step = (int)img.step1();
			int rows = img.rows, cols = img.cols;

			//SIFT_IMG_BORDER=5，边界5个像素的距离
			for (int r = SIFT_IMG_BORDER; r < rows - SIFT_IMG_BORDER; r++)
			{
				//获取3幅相邻的Dog图像的行指针
				const short* currptr = img.ptr<short>(r);
				const short* prevptr = prev.ptr<short>(r);
				const short* nextptr = next.ptr<short>(r);

				for (int c = SIFT_IMG_BORDER; c < cols - SIFT_IMG_BORDER; c++)
				{
					//获取该层图像r行c列的像素值
					int val = currptr[c];

					// find local extrema with pixel accuracy
					//与周围26个点比较，极大或极小值则为局部极值点
					if (std::abs(val) > threshold &&
						((val > 0 && val >= currptr[c - 1] && val >= currptr[c + 1] &&
							val >= currptr[c - step - 1] && val >= currptr[c - step] && val >= currptr[c - step + 1] &&
							val >= currptr[c + step - 1] && val >= currptr[c + step] && val >= currptr[c + step + 1] &&
							val >= nextptr[c] && val >= nextptr[c - 1] && val >= nextptr[c + 1] &&
							val >= nextptr[c - step - 1] && val >= nextptr[c - step] && val >= nextptr[c - step + 1] &&
							val >= nextptr[c + step - 1] && val >= nextptr[c + step] && val >= nextptr[c + step + 1] &&
							val >= prevptr[c] && val >= prevptr[c - 1] && val >= prevptr[c + 1] &&
							val >= prevptr[c - step - 1] && val >= prevptr[c - step] && val >= prevptr[c - step + 1] &&
							val >= prevptr[c + step - 1] && val >= prevptr[c + step] && val >= prevptr[c + step + 1]) ||
							(val < 0 && val <= currptr[c - 1] && val <= currptr[c + 1] &&
								val <= currptr[c - step - 1] && val <= currptr[c - step] && val <= currptr[c - step + 1] &&
								val <= currptr[c + step - 1] && val <= currptr[c + step] && val <= currptr[c + step + 1] &&
								val <= nextptr[c] && val <= nextptr[c - 1] && val <= nextptr[c + 1] &&
								val <= nextptr[c - step - 1] && val <= nextptr[c - step] && val <= nextptr[c - step + 1] &&
								val <= nextptr[c + step - 1] && val <= nextptr[c + step] && val <= nextptr[c + step + 1] &&
								val <= prevptr[c] && val <= prevptr[c - 1] && val <= prevptr[c + 1] &&
								val <= prevptr[c - step - 1] && val <= prevptr[c - step] && val <= prevptr[c - step + 1] &&
								val <= prevptr[c + step - 1] && val <= prevptr[c + step] && val <= prevptr[c + step + 1])))
					{
						//找到极值点之后，在Dog中调整局部极值点
						//调整后的关键点具有以下三个属性
						//在原图中的精确坐标，
						//特征点所在的高斯金字塔组，即更精确的o
						//领域直径
						int r1 = r, c1 = c, layer = i;
						if (!adjustLocalExtrema(dog_pyr, kpt, o, layer, r1, c1,
							nOctaveLayers, (float)contrastThreshold,
							(float)edgeThreshold, (float)sigma))
							continue;

						float scl_octv = kpt.size*0.5f / (1 << o); //获取该特征点的尺度
																	 //calcOrientationHist计算该特征点周围的方向直方图，并返回直方图最大值
																	 //参数o和c1,r1均已经经过精确定位
																	 //方向直方图的计算是在该点尺度的高斯金字塔图像中计算的，不是在Dog图像，也不是在原图
						float omax = calcOrientationHist(gauss_pyr[o*(nOctaveLayers + 3) + layer],
							Point(c1, r1), //特征点坐标
							cvRound(SIFT_ORI_RADIUS * scl_octv), //直方图统计半径：3*1.5*σ，SIFT_ORI_RADIUS=3*1.5
							SIFT_ORI_SIG_FCTR * scl_octv, //直方图平滑所用到的尺度，SIFT_ORI_SIG_FCTR=1.5f
							hist, n);
						float mag_thr = (float)(omax * SIFT_ORI_PEAK_RATIO); //辅方向为0.8*主方向最大值，SIFT_ORI_PEAK_RATIO=0.8f
						for (int j = 0; j < n; j++) //n=36
						{
							int l = j > 0 ? j - 1 : n - 1;
							int r2 = j < n - 1 ? j + 1 : 0;

							if (hist[j] > hist[l] && hist[j] > hist[r2] && hist[j] >= mag_thr)
							{
								float bin = j + 0.5f * (hist[l] - hist[r2]) / (hist[l] - 2 * hist[j] + hist[r2]);
								bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
								kpt.angle = (float)((360.f / n) * bin); //得到关键点的方向
								keypoints.push_back(kpt); //这里保存的特征点具有位置，尺度和方向3个信息
							}
						}
					}
				}
			}
		}
}


void SIFT::calcSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl,
	int d, int n, float* dst)
{
	Point pt(cvRound(ptf.x), cvRound(ptf.y)); //坐标点取整
	float cos_t = cosf(ori*(float)(CV_PI / 180)); //余弦值
	float sin_t = sinf(ori*(float)(CV_PI / 180)); //正弦值
	float bins_per_rad = n / 360.f;
	float exp_scale = -1.f / (d * d * 0.5f);
	float hist_width = SIFT_DESCR_SCL_FCTR * scl;
	int radius = cvRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
	cos_t /= hist_width;
	sin_t /= hist_width;

	int i, j, k, len = (radius * 2 + 1)*(radius * 2 + 1), histlen = (d + 2)*(d + 2)*(n + 2);
	int rows = img.rows, cols = img.cols;

	AutoBuffer<float> buf(len * 6 + histlen);
	float *X = buf, *Y = X + len, *Mag = Y, *Ori = Mag + len, *W = Ori + len;
	float *RBin = W + len, *CBin = RBin + len, *hist = CBin + len;

	//初始化直方图
	for (i = 0; i < d + 2; i++)
	{
		for (j = 0; j < d + 2; j++)
			for (k = 0; k < n + 2; k++)
				hist[(i*(d + 2) + j)*(n + 2) + k] = 0.;
	}

	for (i = -radius, k = 0; i <= radius; i++)
		for (j = -radius; j <= radius; j++)
		{

			float c_rot = j * cos_t - i * sin_t;
			float r_rot = j * sin_t + i * cos_t;
			float rbin = r_rot + d / 2 - 0.5f;
			float cbin = c_rot + d / 2 - 0.5f;
			int r = pt.y + i, c = pt.x + j;

			if (rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
				r > 0 && r < rows - 1 && c > 0 && c < cols - 1)
			{
				
				float dx = (float)(img.at<short>(r, c + 1) - img.at<short>(r, c - 1));
				float dy = (float)(img.at<short>(r - 1, c) - img.at<short>(r + 1, c));
				X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
				W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;
				k++;
			}
		}

	len = k;
	fastAtan2(Y, X, Ori, len, true);
	magnitude(X, Y, Mag, len);
	exp(W, W, len);

	for (k = 0; k < len; k++)
	{
		float rbin = RBin[k], cbin = CBin[k];
		float obin = (Ori[k] - ori)*bins_per_rad;
		float mag = Mag[k] * W[k];

		int r0 = cvFloor(rbin);
		int c0 = cvFloor(cbin);
		int o0 = cvFloor(obin);
		rbin -= r0;
		cbin -= c0;
		obin -= o0;

		if (o0 < 0)
			o0 += n;
		if (o0 >= n)
			o0 -= n;

		// histogram update using tri-linear interpolation
		float v_r1 = mag * rbin, v_r0 = mag - v_r1;
		float v_rc11 = v_r1 * cbin, v_rc10 = v_r1 - v_rc11;
		float v_rc01 = v_r0 * cbin, v_rc00 = v_r0 - v_rc01;
		float v_rco111 = v_rc11 * obin, v_rco110 = v_rc11 - v_rco111;
		float v_rco101 = v_rc10 * obin, v_rco100 = v_rc10 - v_rco101;
		float v_rco011 = v_rc01 * obin, v_rco010 = v_rc01 - v_rco011;
		float v_rco001 = v_rc00 * obin, v_rco000 = v_rc00 - v_rco001;

		int idx = ((r0 + 1)*(d + 2) + c0 + 1)*(n + 2) + o0;
		hist[idx] += v_rco000;
		hist[idx + 1] += v_rco001;
		hist[idx + (n + 2)] += v_rco010;
		hist[idx + (n + 3)] += v_rco011;
		hist[idx + (d + 2)*(n + 2)] += v_rco100;
		hist[idx + (d + 2)*(n + 2) + 1] += v_rco101;
		hist[idx + (d + 3)*(n + 2)] += v_rco110;
		hist[idx + (d + 3)*(n + 2) + 1] += v_rco111;
	}

	// finalize histogram, since the orientation histograms are circular
	for (i = 0; i < d; i++)
		for (j = 0; j < d; j++)
		{
			int idx = ((i + 1)*(d + 2) + (j + 1))*(n + 2);
			hist[idx] += hist[idx + n];
			hist[idx + 1] += hist[idx + n + 1];
			for (k = 0; k < n; k++)
				dst[(i*d + j)*n + k] = hist[idx + k];
		}
	// copy histogram to the descriptor,
	// apply hysteresis thresholding
	// and scale the result, so that it can be easily converted
	// to byte array
	float nrm2 = 0;
	len = d * d*n;
	for (k = 0; k < len; k++)
		nrm2 += dst[k] * dst[k];
	float thr = std::sqrt(nrm2)*SIFT_DESCR_MAG_THR;
	for (i = 0, nrm2 = 0; i < k; i++)
	{
		float val = std::min(dst[i], thr);
		dst[i] = val;
		nrm2 += val * val;
	}
	nrm2 = SIFT_INT_DESCR_FCTR / std::max(std::sqrt(nrm2), FLT_EPSILON);
	for (k = 0; k < len; k++)
	{
		dst[k] = std::sqrt(dst[k] * nrm2);
	}
}

void SIFT::calcDescriptors(const vector<Mat>& gpyr, const vector<KeyPoint>& keypoints,
	Mat& descriptors, int nOctaveLayers)
{
	//SIFT_DESCR_WIDTH=4，描述直方图的宽度
	//SIFT_DESCR_HIST_BINS=8
	int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;

	for (size_t i = 0; i < keypoints.size(); i++)
	{
		KeyPoint kpt = keypoints[i];
		int octv = kpt.octave & 255, layer = (kpt.octave >> 8) & 255; //该特征点所在的组序号和层序号
		float scale = 1.f / (1 << octv); //缩放倍数
		float size = kpt.size*scale; //该特征点所在组的图像尺寸
		Point2f ptf(kpt.pt.x*scale, kpt.pt.y*scale); //该特征点在金字塔组中的坐标
		const Mat& img = gpyr[octv*(nOctaveLayers + 3) + layer]; //该点所在的金字塔图像

		calcSIFTDescriptor(img, ptf, kpt.angle, size*0.5f, d, n, descriptors.ptr<float>((int)i));
	}
}

int SIFT::descriptorSize() const
{
	return SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS; //4*4*8
}

int SIFT::descriptorType() const
{
	return CV_32F;
}

void SIFT::detectImpl(const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask) const
{
	(*this)(image, mask, keypoints, noArray());
}

void SIFT::computeImpl(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) const
{
	(*this)(image, Mat(), keypoints, descriptors, true);
}


//using namespace cv;




int _tmain(int argc, _TCHAR* argv[])
{
	//Mat m(2,2,CV_8UC1,10);
	//std::cout<<"m="<<m<<std::endl;
	//Mat m_con;
	//m.convertTo(m_con,CV_8UC1,0.5,0);   //m_con=alpha*m+beta;
	//std::cout<<"m_con"<<m_con<<std::endl;

	Mat img = imread("D://1.jpg");
	std::cout << "原始lena图像大小：" << img.size().width << "*" << img.size().height << std::endl;
	std::cout << "原始lena图像通道数：" << img.channels() << std::endl;
	std::cout << "原始lena图像数值类型（0表示每个通道为8位UC类型）：" << img.depth() << std::endl << std::endl;
	//Mat gray;
	//cvtColor(img,gray,COLOR_BGR2GRAY);
	//imshow("gray",gray);
	//imwrite("gray.jpg",gray);
	//Mat gray_fpt;
	//gray.convertTo(gray_fpt,CV_16S,SIFT_FIXPT_SCALE,0);
	//imshow("gray_fpt",gray_fpt);
	// imwrite("gray_fpt.jpg",gray_fpt);  //不能保存CV_16S深度的图像？


	//创建初始图像
	//bool doubleImageSize=false;
	//double sigma=1.6;
	SIFT sift;
	Mat base;
	base = sift.createInitialImage(img, false, sift.sigma); //base由gray_fpt经过高斯模糊后得到
															 //imshow("base",base);
	std::cout << "初始图像大小：" << base.size().width << "*" << base.size().height << std::endl;
	std::cout << "初始图像通道数：" << base.channels() << std::endl;
	std::cout << "初始图像数值类型（3表示每个通道为16位SC类型）：" << base.depth() << std::endl << std::endl;


	vector<Mat> gpyr, dogpyr;
	////高斯金字塔的组数
	int nOctaves = cvRound(log((double)std::min(base.cols, base.rows)) / log(2.) - 2);
	std::cout << "高斯金字塔的组数nOctaves=" << nOctaves << std::endl << std::endl;
	//int nOctaves=  cvRound(log( (double)std::min( base.cols,base.rows) )/log(2.)-2);

	//构造高斯金字塔
	sift.buildGaussianPyramid(base, gpyr, nOctaves);
	//保存金字塔图像，因为金字塔图像灰度扩大了48倍，所以保存时要缩小48倍才能看到图像，否则为一片白色

	//构造差分金字塔
	sift.buildDoGPyramid(gpyr, dogpyr);
	//保存Dog金字塔图像

	//找到特征点并去除重复特征点
	vector<KeyPoint> keypoints;
	sift.findScaleSpaceExtrema(gpyr, dogpyr, keypoints);
	KeyPointsFilter::removeDuplicated(keypoints);

	//  for (int i=0;i
	//  {
	// fout<<keypoints[i].pt<<'\n';
	////  std::cout<<keypoints[i].pt<<" ";
	//  }

	//保留指定数目的关键点
	if (sift.nfeatures > 0)
		KeyPointsFilter::retainBest(keypoints, sift.nfeatures);


	//计算描述符
	int dsize = sift.descriptorSize(); //128
	Mat descriptors;
	descriptors.create((int)keypoints.size(), dsize, CV_32F);
	sift.calcDescriptors(gpyr, keypoints, descriptors, sift.nOctaveLayers);


	Mat img_keypoints;
	drawKeypoints(img, keypoints, img_keypoints, cv::Scalar::all(-1), 0);
	imwrite("Lena_keypoints.jpg", img_keypoints);
	imshow("show", img_keypoints);
	

	waitKey();
	return 0;
}

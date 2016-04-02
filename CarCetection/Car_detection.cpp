#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>
using namespace std;
using namespace cv;
int detectAndDraw( Mat& img, CascadeClassifier& cascade, double scale, bool tryflip ,int num);
string cascadeName = "D:\\Program Files\\opencv\\CV_project\\CarDetection\\CarDetection\\LBPcascade.xml";
//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
int main()
{
    Mat  image;
	int num,number;
	char Filname[25];
	num=267;
    string inputName;
    bool tryflip = false;
    CascadeClassifier cascade, nestedCascade;
    double scale = 1;
    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        return -1;
    }
	cvNamedWindow( "result", 0 );
	for(int number=7700;number<7713;number++)
	{
		sprintf(Filname, "%s%d%s", "G:\\HaarTraining\\Neg\\", number, ".jpg");//保存的图片名
		image = imread( Filname,1 );
		cout << "In image read" << endl;
		if( !image.empty() )
		{
			num=detectAndDraw( image, cascade,  scale, tryflip,num);
			//waitKey(0);
		}
	}

    cvDestroyWindow("result");
    return 0;
}
//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
int detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    double scale, bool tryflip,int num )
{
    int i = 0;
    double t = 0;
	char image_name[25];
    vector<Rect> faces, faces2;
    const static Scalar colors= CV_RGB(128,255,0);
    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );
    cvtColor( img, gray, CV_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );
    t = (double)cvGetTickCount();
    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        //|CV_HAAR_FIND_BIGGEST_OBJECT
        //|CV_HAAR_DO_ROUGH_SEARCH
        |CV_HAAR_SCALE_IMAGE,
        Size(30, 30) );
    if( tryflip )
    {
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale( smallImg, faces2,
                                 1.1, 2, 0
                                 //|CV_HAAR_FIND_BIGGEST_OBJECT
                                 //|CV_HAAR_DO_ROUGH_SEARCH
                                 |CV_HAAR_SCALE_IMAGE,
                                 Size(30, 30) );
        for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++ )
        {
            faces.push_back(cvRect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    t = (double)cvGetTickCount() - t;
    printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
    {
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        int radius;
        double aspect_ratio = (double)r->width/r->height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
			num=num+1;
            //center.x = cvRound((r->x + r->width*0.5)*scale);
            //center.y = cvRound((r->y + r->height*0.5)*scale);
            //radius = cvRound((r->width + r->height)*0.25*scale);
			Mat srcROI=img(cvRect(r->x,r->y,r->width,r->height));
			sprintf(image_name, "%s%d%s", "G:\\OpenCV\\data8\\", num, ".jpg");//保存的图片名
			imwrite(image_name,srcROI);
			//rectangle( img,cvPoint(center.x-radius,center.y-radius),cvPoint(center.x+radius,center.y+radius),colors, 3, 8, 0);
        }
        else
            rectangle( img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                       cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                       colors, 3, 8, 0);
		//IplImage*out=cvCreateImage(cvSize(r->width,r->height),IPL_DEPTH_8U,3);

    }
    //cv::imshow( "result", img );
	return num;
}

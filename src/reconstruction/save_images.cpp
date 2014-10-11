#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/warpers.hpp>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cwchar>
#include <string>

#include "Magic.h"
#include "Canvas.h"

using namespace cv;
using namespace std;

int main() {
	VideoCapture cam1;
	cam1.open(2);

	VideoCapture cam2;
	cam2.open(1);

	int i = 1;
	while (true) {
		cam1.grab();
		cam2.grab();

		Mat img1, img2,img1_, img2_;
		cam1.read(img1);
		cam2.read(img2);

		imshow("cam1", img1);
		imshow("cam2", img2);

		int c = waitKey(50);
		if (c == ' ') {
			ostringstream convert;
			convert << "left" << ((i < 10) ? "0" : "") << i << ".jpg";
			imwrite(convert.str(), img1);
			convert.str("");
			convert << "right" << ((i < 10) ? "0" : "") << i << ".jpg";
			imwrite(convert.str(), img2);
			i++;
		} else if (c == 27) {
			break;
		}
	}

	return 0;
}

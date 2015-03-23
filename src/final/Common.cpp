//
// Created by balint on 2015.03.22..
//

#include "Common.h"

#include <fstream>

using namespace cv;
using namespace std;

vector<Point3d> CloudPointsToPoints(const vector<CloudPoint>& cpts) {
    vector<Point3d> out;
    for (unsigned int i=0; i<cpts.size(); i++) {
        out.push_back(cpts[i].pt);
    }
    return out;
}

void writeCloudPoints(const vector<CloudPoint>& cpts) {
    ofstream myfile;

    myfile.open ("/media/balint/Data/Linux/cloud.ply");
    myfile << "ply " << endl;
    myfile << "format ascii 1.0" << endl;
    myfile << "element vertex " << cpts.size() << endl;
    myfile << "property float x" << endl;
    myfile << "property float y" << endl;
    myfile << "property float z" << endl;
    myfile << "end_header" << endl;
    for (unsigned int i=0; i<cpts.size(); i++) {
        myfile << cpts[i].pt.x << " " << cpts[i].pt.y << " " << cpts[i].pt.z << endl;
    }
    myfile.close();
}
#include <opencv2/core/utility.hpp>
#include <omp.h>
#include "PerformanceMonitor.h"

PerformanceMonitor *PerformanceMonitor::instance = NULL;

PerformanceMonitor *PerformanceMonitor::get() {
    if (instance == NULL) {
        instance = new PerformanceMonitor();
    }

    return instance;
}

double PerformanceMonitor::getTimeSince(int64 started) {
    return (double) (cv::getTickCount() - started) / cv::getTickFrequency();
}

void PerformanceMonitor::frameStarted() {
    _frameStarted = cv::getTickCount();
}

void PerformanceMonitor::frameFinished() {
    frameTotal.addNew(getTimeSince(_frameStarted));
}

void PerformanceMonitor::maskCalculationStarted() {
    _maskCalculationStarted = cv::getTickCount();
}

void PerformanceMonitor::maskCalculationFinished() {
    maskCalculation.addNew(getTimeSince(_maskCalculationStarted));
}

void PerformanceMonitor::extractingStarted() {
    _extractingStarted = cv::getTickCount();
}

void PerformanceMonitor::extractingFinished() {
    extracting.addNew(getTimeSince(_extractingStarted));
}

void PerformanceMonitor::objSelectionStarted() {
    _objSelectionStarted = cv::getTickCount();
}

void PerformanceMonitor::objSelectionFinished() {
    objSelection.addNew(getTimeSince(_objSelectionStarted));
}

void PerformanceMonitor::objFound(unsigned long i) {
    objectCount[i]++;
}

void PerformanceMonitor::ofInitStarted() {
    _ofInitStarted[omp_get_thread_num()] = cv::getTickCount();
}

void PerformanceMonitor::ofInitFinished() {
    double duration = getTimeSince(_ofInitStarted[omp_get_thread_num()]);
    ofInit[omp_get_thread_num()].addNew(duration);
#pragma omp atomic
    ofInitPerFrameDuration += duration;
}

void PerformanceMonitor::ofCalcStarted() {
    _ofCalcStarted[omp_get_thread_num()] = cv::getTickCount();
}

void PerformanceMonitor::ofCalcFinished() {
    double duration = getTimeSince(_ofCalcStarted[omp_get_thread_num()]);
    ofCalc[omp_get_thread_num()].addNew(duration);
#pragma omp atomic
    ofCalcPerFrameDuration += duration;
}

void PerformanceMonitor::ofMatchingStarted() {
    _ofMatchingStarted[omp_get_thread_num()] = cv::getTickCount();
}

void PerformanceMonitor::ofMatchingFinished() {
    double duration = getTimeSince(_ofMatchingStarted[omp_get_thread_num()]);
    ofMatching[omp_get_thread_num()].addNew(duration);
#pragma omp atomic
    ofMatchingPerFrameDuration += duration;
}

void PerformanceMonitor::triangulationStarted() {
    _triangulationStarted[omp_get_thread_num()] = cv::getTickCount();
}

void PerformanceMonitor::triangulationFinished() {
    double duration = getTimeSince(_triangulationStarted[omp_get_thread_num()]);
    triangulation[omp_get_thread_num()].addNew(duration);
#pragma omp atomic
    triangulationPerFrameDuration += duration;
}

void PerformanceMonitor::objectBasedStuff() {
    ofInitPerFrame.addNew(ofInitPerFrameDuration);
    ofCalcPerFrame.addNew(ofCalcPerFrameDuration);
    ofMatchingPerFrame.addNew(ofMatchingPerFrameDuration);
    triangulationPerFrame.addNew(triangulationPerFrameDuration);

    ofInitPerFrameDuration = 0.0;
    ofCalcPerFrameDuration = 0.0;
    ofMatchingPerFrameDuration = 0.0;
    triangulationPerFrameDuration = 0.0;
}

void PerformanceMonitor::visualizationStarted() {
    _visualizationStarted = cv::getTickCount();
}

void PerformanceMonitor::visualizationFinished() {
    visualization.addNew(getTimeSince(_visualizationStarted));
}

void PerformanceMonitor::reset() {
    frameTotal.reset();
    maskCalculation.reset();
    extracting.reset();
    objSelection.reset();
    ofInit = std::map<int, PerformanceIndicator>();
    ofCalc = std::map<int, PerformanceIndicator>();
    ofMatching = std::map<int, PerformanceIndicator>();
    triangulation = std::map<int, PerformanceIndicator>();
    visualization.reset();
    ofInitPerFrame.reset();
    ofCalcPerFrame.reset();
    ofMatchingPerFrame.reset();
    triangulationPerFrame.reset();
}

void PerformanceMonitor::reprojError(double d) {
#pragma omp critical
    reproj.addNew(d);
}

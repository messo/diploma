#pragma once

#include <map>
#include "PerformanceIndicator.h"

class PerformanceMonitor {

private:
    PerformanceMonitor() { }

    static PerformanceMonitor *instance;

    double getTimeSince(int64 started);

public:

    static PerformanceMonitor *get();

    void frameStarted();

    void frameFinished();

    int64 _frameStarted;
    PerformanceIndicator frameTotal;

    void maskCalculationStarted();

    void maskCalculationFinished();

    int64 _maskCalculationStarted;
    PerformanceIndicator maskCalculation;

    void extractingStarted();

    void extractingFinished();

    int64 _extractingStarted;
    PerformanceIndicator extracting;

    void objSelectionStarted();

    void objSelectionFinished();

    int64 _objSelectionStarted;
    PerformanceIndicator objSelection;

    void objFound(unsigned long i);

    std::map<unsigned long, int> objectCount;

    void ofInitStarted();

    void ofInitFinished();

    int64 _ofInitStarted;
    PerformanceIndicator ofInit;

    void ofCalcStarted();

    void ofCalcFinished();

    int64 _ofCalcStarted;
    PerformanceIndicator ofCalc;

    void ofMatchingStarted();

    void ofMatchingFinished();

    int64 _ofMatchingStarted;
    PerformanceIndicator ofMatching;

    void triangulationStarted();

    void triangulationFinished();

    int64 _triangulationStarted;
    PerformanceIndicator triangulation;

    void visualizationStarted();

    void visualizationFinished();

    int64 _visualizationStarted;
    PerformanceIndicator visualization;

    void objectBasedStuff();

    PerformanceIndicator ofInitPerFrame;
    PerformanceIndicator ofCalcPerFrame;
    PerformanceIndicator ofMatchingPerFrame;
    PerformanceIndicator triangulationPerFrame;

    double ofInitPerFrameDuration = 0.0;
    double ofCalcPerFrameDuration = 0.0;
    double ofMatchingPerFrameDuration = 0.0;
    double triangulationPerFrameDuration = 0.0;
};


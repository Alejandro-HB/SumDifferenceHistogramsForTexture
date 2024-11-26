#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <numeric>
#include <cmath>

using namespace std;
using namespace cv;

void hbComputeHistograms(const Mat& image, vector<int>& sumHistogram, vector<int>& diffHistogram, int distance, double angle);
void hbShowImageHistogram(string winTitle, const vector<int>& histogram, int histSize, vector<double>&normalizedHist);
double hbComputeMean(const vector<double>& sumHistogram);
double hbComputeEnergy(const vector<double>& sumHistogram, const vector<double>& diffHistogram);
double hbComputeHomogeneity(const vector<double>& diffHistogram);
double hbComputeContrast(const vector<double>& diffHistogram);
double hbComputeEntropy(const vector<double>& sumHistogram, const vector<double>& diffHistogram);

int main() {

    //string filename = "../../../../cvImagesNotMed/img11.jpg";
    //string filename = "./img/GRIS-GRAFITO-S277.png";
    string filename = "./img/Periodic-texture-mapping-versus-aperiodic-texture-mapping-The-texture-on-the-left-was.png";
    //string filename = "./img/bright-river.png";
    Mat inputImage = imread(filename, IMREAD_GRAYSCALE);
    if (!inputImage.data) {
        cout << "\nUnable to read input image" << endl;
        return -1;
    }

    namedWindow("Input image", WINDOW_NORMAL);
    imshow("Input image", inputImage);

    vector<int> sumHistogram, diffHistogram;
    vector<double> normalizedSumHistogram, normalizedDiffHistogram;

    vector<double> jointSumHistogram(511, 0.0);  //Combined histogram for sum
    vector<double> jointDiffHistogram(511, 0.0); //Combined histogram for difference


    vector<int> distances = {1, 2};
    vector<int> angles = {0, 45, 90, 135};

    for (int distance : distances) {
        for (double angle : angles) {
            cout << "\nDistance = " << distance << ", angle = " << angle << "°\n";

            //Compute histograms
            hbComputeHistograms(inputImage, sumHistogram, diffHistogram, distance, angle);

            //Display histograms
            hbShowImageHistogram("Sum Histogram", sumHistogram, 511, normalizedSumHistogram);
            hbShowImageHistogram("Difference Histogram", diffHistogram, 511, normalizedDiffHistogram);

            //Combine normalized histograms
            for (int i = 0; i < 511; i++) {
                jointSumHistogram[i] += normalizedSumHistogram[i];
                jointDiffHistogram[i] += normalizedDiffHistogram[i];
            }

            //Compute statistical properties
            double mean = hbComputeMean(normalizedSumHistogram);
            double energy = hbComputeEnergy(normalizedSumHistogram, normalizedDiffHistogram);
            double contrast = hbComputeContrast(normalizedDiffHistogram);
            double homogeneity = hbComputeHomogeneity(normalizedDiffHistogram);
            double entropy = hbComputeEntropy(normalizedSumHistogram, normalizedDiffHistogram);

            //Show results
            cout << "\nMean: " << mean << endl;
            cout << "Energy: " << energy << endl;
            cout << "Contrast: " << contrast << endl;
            cout << "Homogeneity: " << homogeneity << endl;
            cout << "Entropy: " << entropy << endl;
        }
    }


    //Final result
    //Normalize combined histograms for having a PDF
    double totalSum = accumulate(jointSumHistogram.begin(), jointSumHistogram.end(), 0.0);
    double totalDiff = accumulate(jointDiffHistogram.begin(), jointDiffHistogram.end(), 0.0);

    for (int i = 0; i < 511; i++) {
        jointSumHistogram[i] /= totalSum;
        jointDiffHistogram[i] /= totalDiff;
    }

    //Compute features using the combined histograms
    double mean = hbComputeMean(jointSumHistogram);
    double energy = hbComputeEnergy(jointSumHistogram, jointDiffHistogram);
    double contrast = hbComputeContrast(jointDiffHistogram);
    double homogeneity = hbComputeHomogeneity(jointDiffHistogram);
    double entropy = hbComputeEntropy(jointSumHistogram, jointDiffHistogram);

    //Show final results
    cout << "\nFinal Results:\n";
    cout << "Mean: " << mean << endl;
    cout << "Energy: " << energy << endl;
    cout << "Contrast: " << contrast << endl;
    cout << "Homogeneity: " << homogeneity << endl;
    cout << "Entropy: " << entropy << endl;

    waitKey(0);
    return 0;
}

void hbComputeHistograms(const Mat& image, vector<int>& sumHistogram, vector<int>& diffHistogram, int distance, double angle) {
    int maxGrayLevel = 255;
    int sumHistogramSize = 2 * maxGrayLevel + 1; // Sum range: [0, 510]
    int diffHistogramSize = sumHistogramSize;    // Diff range: [0, 510]

    sumHistogram.assign(sumHistogramSize, 0);
    diffHistogram.assign(diffHistogramSize, 0);

    // Compute diferences based on angle
    int dx = 0, dy = 0;
    if (angle == 0) {
        dx = distance;
        dy = 0;
    } else if (angle == 45) {
        dx = distance;
        dy = -distance;
    } else if (angle == 90) {
        dx = 0;
        dy = -distance;
    } else if (angle == 135) {
        dx = -distance;
        dy = -distance;
    }

    //Move through the image
    for (int i = max(0, -dy); i < image.rows - max(0, dy); i++) {
        for (int j = max(0, -dx); j < image.cols - max(0, dx); j++) {
            int currentPixel = image.at<uchar>(i, j);
            int neighborPixel = image.at<uchar>(i + dy, j + dx);

            //Compute sum and difference
            int sum = currentPixel + neighborPixel;
            int diff = currentPixel - neighborPixel;

            //Update histograms
            sumHistogram[sum]++;
            diffHistogram[diff + 255]++;
        }
    }

    //Total occurrences
    int totalOccurrencesSum = accumulate(sumHistogram.begin(), sumHistogram.end(), 0);
    int totalOccurrencesDiff = accumulate(diffHistogram.begin(), diffHistogram.end(), 0);
    cout << "Sum occurrences: " << totalOccurrencesSum << endl;
    cout << "Diff occurrences: " << totalOccurrencesDiff << endl;
}




void hbShowImageHistogram(string winTitle, const vector<int>& histogram, int histSize, vector<double>&normalizedHist) {
    //Configuration
    int hist_w = 511, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    // Create histogram image
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(128, 128, 128)); // Gray background


    //Resize normalized histogram
    normalizedHist.resize(histSize, 0.0);
    //Compute total occurrences in the histogram
    int totalOccurrences = accumulate(histogram.begin(), histogram.end(), 0);
    double prob=0;
        for (int i = 0; i < histSize; i++) {
            normalizedHist[i] = (double)histogram[i] / totalOccurrences;
            prob+=normalizedHist[i];
        }
    cout<<"Prob: "<<prob<<endl;
    //else
        //for (int i = 0; i < histSize; i++) {
            //normalizedHist[i] = (double)histogram[i];
        //}
    // Scale normalized values to fit within the image height
    vector<int> scaledHist(histSize);
    for (int i = 0; i < histSize; i++) {
        scaledHist[i] = cvRound(normalizedHist[i] * hist_h);
    }

    // Draw the histogram
    for (int i = 1; i < histSize; i++) {
        line(histImage,
             Point(bin_w * (i - 1), hist_h - scaledHist[i - 1]),
             Point(bin_w * i, hist_h - scaledHist[i]),
             Scalar(0, 255, 0), 2, 8, 0);
    }

    // Display the histogram
    namedWindow(winTitle, WINDOW_NORMAL);
    imshow(winTitle, histImage);
    string filename = winTitle+"histogram.png";
    imwrite("./img/periodic/"+filename, histImage);
    //return normalizedHist;
}


double hbComputeMean(const vector<double>& sumHistogram){
    double sum=0;
    double u=0;

    for (int i=0; i<sumHistogram.size(); i++) {
        sum+=(i*sumHistogram[i]);
    }

    u=sum/2;
    return u;
}

double hbComputeEnergy(const vector<double>& sumHistogram, const vector<double>& diffHistogram){
    double sumSum=0;
    double sumDiff=0;
    double energy=0;

    for(int i=0; i<sumHistogram.size();i++){
        sumSum+=pow(sumHistogram[i],2);
    }

    
    for(int j=0; j<diffHistogram.size(); j++){
        sumDiff+=pow(diffHistogram[j],2);
    }

    energy=sumSum*sumDiff;
    return energy;
}

double hbComputeContrast(const vector<double>& diffHistogram){
    double sum=0;

    for(int j=0; j<diffHistogram.size(); j++){
        sum+=(pow(j-255,2)*diffHistogram[j]);
    }

    return sum;
}

double hbComputeHomogeneity(const vector<double>& diffHistogram) {
    double sum=0;
    for(int j=0; j<diffHistogram.size(); j++){
        sum+=(diffHistogram[j]/((1+pow(j-255,2))));
    }

    return sum;
}

double hbComputeEntropy(const vector<double>& sumHistogram, const vector<double>& diffHistogram){
    double entropy=0;
    double sumSum=0;
    double sumDiff=0;

    for(int i=0; i<sumHistogram.size(); i++){
        if (sumHistogram[i] == 0)
            sumSum+=(sumHistogram[i]*log(1e-10));
        else
            sumDiff+=(sumHistogram[i]*log(sumHistogram[i]));

    }

    for(int j=0; j<diffHistogram.size(); j++){
        if (diffHistogram[j] == 0)
            sumDiff+=(diffHistogram[j]*log(1e-10));
        else
            sumDiff+=(diffHistogram[j]*log(diffHistogram[j]));
    }

    entropy=-sumSum-sumDiff;
    return entropy;
}



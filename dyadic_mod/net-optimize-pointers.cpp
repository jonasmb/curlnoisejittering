/*
 * Generates optimized (0, m, 2)-nets, as described in the paper:
 *      Ahmed and Wonka: "Optimizing Dyadic Nets"
 *
 * Compilation:
 *      g++ -O3 -march=core2 -msse3 -no-pie -o net-optimize-pointers net-optimize-pointers.cpp
 * Execution example:
 *      ./net-optimize-pointers -q v -s 0.5 -n 1000 1024
 * This produces 1024 points, optimizes them using Gaussian BN, up to 1000 iterations.
 *
 * 2020-06-07: First created by Abdalla Ahmed.
 * 2021-05-04: Revised by Abdalla Ahmed for inclusion with the paper.
 */

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <time.h>
#include <string>
#include <cstring>
#include <getopt.h>
#include <signal.h>
#include <algorithm>
#include <stdint.h>

struct Point {
    int x, y;
};
typedef std::vector<Point> Points;
//typedef std::vector<uint32_t> List;                                             // A linear list of uint32_t integers
typedef std::vector<int> List;    // jonas: I had to change from uint32_t to int to be able to compile                                         
typedef std::string String;

const double dhex = 1.07456993182354;                                           // sqrt(2/sqrt(3)).
static bool interuptFlag = false;

inline void shuffle(List &list) {                                               // Populate a randomly ordered list.
    uint32_t N = list.size();
    for (uint32_t i = 0; i < N; i++) list[i] = i;
    for (uint32_t i = N - 1; i > 0; i--) {                                      // Iterate down the list, and swap each place with another place down, inclusive of same place.
        uint32_t r = rand() % (i + 1);                                          // (i + 1) means that the same place is included.
        std::swap(list[i], list[r]);
    }
}

class Net {
private:
    int N;                                                                      // The number of points, must be a power of 2.
    int half;                                                                   // Shorthand for N / 2.
    int m;                                                                      // log_2(N).
    bool useOwen;                                                               // If the net is a sequence, optimization needs to be restricted to Own's scrambling.
    Points p;                                                                   // List of points
    std::vector<List> q;                                                        // Back reference from points to strata
    int main;                                                                   // Index of stratification used for neighbor search.
    int width, widthBits;                                                       // Number of strata in a row of main, which is always row-major.
    int height, heightBits;                                                     // Number of strata in a column of main.
    int y2x;                                                                    // Used if the main strata are not square, when m is odd.
    std::vector<double> kernel;                                                 // Gaussian kernel for void-and-cluster optimization.
    double sigma, sigmaSq2Inv;                                                  // SD of Gaussian kernel.
    double rfSq;                                                                // Square of target conflict radius.
    double conflictRadiusFactor;
    uint32_t filterRange;                                                       // Range of Gaussian filter.
    int range;                                                                  // Neighborhood to consider in cluster optimization.
    void init();
    List netSort();                                                             // Return a (0, 1)-sequence.
    uint32_t p2q(uint32_t i, uint32_t k);                                       // Retrieve the stratum of point i in the k'th stratification
    void swap(uint32_t k, uint32_t stratum);
    uint32_t torroidalDistance(uint32_t x1, uint32_t x2);
    uint32_t getConflictRadiusSq(uint32_t i);                                   // Conflict radius of the indexed point
    int maxConflict(uint32_t i, int k);                                         // Swap the i'th point width the candidate in the k'th stratification to maximize conflict radius. Return 1 if swap is accepted, 0 otherwise.
    int minConflict(uint32_t i, int k);
    double g(int dx);                                                           // Gaussian, accounting for periods
    double cluster(uint32_t i);                                                 // Compute the Gaussian-weighted distance of other points at the point location in a given stratum, for void-and-cluster filtering.
    int minCluster(uint32_t i, int k);                                          // Swap the i'th point width the candidate in the k'th stratification to reduce the cluster tightness. Return 1 if swap is accepted, 0 otherwise.
    String outputPath;                                                          // Path for output files; e.g. a sub-folder under /tmp
    int (Net::*energyFunction) (uint32_t i, int k);                             // A pointer to a function for energy-based swapping.
    int optimize(int i);                                                        // Make all permissible swaps to optimize relative to a given point index, using the designated energy function, and return the number of applied swaps.
public:
    Net(int pointCount, String path = "");                                      // Create an N-points binary net, where N is a power of 2.
    Net(char *pbmFileName, String path = "");                                   // Generate the net from stored flags
    void printText(String fileName = "net.txt");                                // Print a .txt file to stdout.
    void printEPS(String fileName = "net.eps");                                 // Print a paginated EPS file showing elementary intervals.
    void printFlags(String fileName = "net.pbm");                               // Print a bitmap of flags.
    void printComplement(String fileName = "complement.txt");
    void setSigma(double v);
    void setRf(double v);
    void setRange(int v);                                                       // Set neighborhood for cluster optimization; default is n2/2.
    double conflictRadius();                                                    // Global conflict radius.
    void optimize(std::string seq, int iterations);
    void upsample();
};

Net::Net(int pointCount, String path) {
    outputPath = path;
    m = ceil(std::log2(pointCount));                                            // Round up to a power of 2.
    N = 1 << m;
    p.resize(N);
    List xlist = netSort();                                                     // Populate the y's with a (0, m, 1)-net
    for (int i = 0; i < N; i++) {
        p[i] = {xlist[i], i};
    }
    init();
}

Net::Net(char *pbmFileName, String path) {
    outputPath = path;
    FILE *pbmFile = fopen(pbmFileName, "r");
    fscanf(pbmFile, "P1 %d %d", &N, &m);
    N *= 2;
    if (N != (1 << m)) {
        fprintf(stderr, "Wrong flags table size\n");
        exit(1);
    }
    p.resize(N);
    List xlist(N), tmpList(N);                                                  // We maintain two buffers and swap them
    for (int i = 0; i < N; i++) xlist[i] = i;                                   // Initialize to a natural order
    for (uint32_t span = N; span > 1; span >>= 1) {                             // Size of sorted sub sets, starting at the whole set
        for (uint32_t slotNo = 0; slotNo < N; slotNo += 2) {                    // Iterate through pairs of slots
            uint32_t relevantBits = slotNo & (span - 1);                        // The set will be permuted in this range only
            uint32_t newSlot0 = slotNo - relevantBits + (relevantBits >> 1);    // Perform a bit rotate over the relevant bits; the least significant being 0;
            uint32_t newSlot1 = newSlot0 + (span >> 1);                         // This is the partial bit rotate for slotNo + 1
            char c;
            fscanf(pbmFile, " %c", &c);
            uint32_t toggle = c == '0' ? 0 : 1;
            tmpList[newSlot0] = xlist[slotNo ^ toggle];
            tmpList[newSlot1] = xlist[(slotNo + 1) ^ toggle];
        }
        std::swap(xlist, tmpList);
    }
    for (int i = 0; i < N; i++) {
        p[i] = {xlist[i], i};
    }
    init();
}

void Net::init() {
    half = N / 2;
    main = m / 2;                                                               // Main stratification used for neighbor search
    y2x = (m & 1) ? 1 : 2;                                                      // Aspect ratio of main strata
    widthBits = ((m + 1) / 2);
    heightBits = m - widthBits;
    width = 1 << widthBits;
    height = N >> widthBits;
    conflictRadiusFactor = 1 / (sqrt(N) * dhex);
    fprintf(
        stderr, "N = %d, m = %d, main is %d, width = %d,"
        " height = %d, y2x = %d\n",
        N, m, main, width, height, y2x
    );
    q.resize(m+1);
    for (int k = 0; k <= m; k++) {
        q[k].resize(N);
    }
    for (int k = 0; k <= m; k++) {
        for (int i = 0; i < N; i++) {
            q[k][p2q(i, k)] = i;                                                // Point strata to points
        }
    }
    //filterRange = half + 1;
    filterRange = std::max(1025,half+1);
    kernel.resize(filterRange);
    setSigma(1.5);
    setRange(0);
    useOwen = false;
}

void Net::swap(uint32_t k_ref, uint32_t stratum) {
    uint32_t i = q[k_ref][stratum];
    uint32_t j = q[k_ref][stratum ^ 1];
    std::swap(p[i].y, p[j].y);                                                  // We always keep the x unchanged
    for (int k = k_ref + 1; k <= m; k++) {                                      // Affected stratifications
        std::swap(q[k][ p2q(i,k) ], q[k][ p2q(j,k) ]);
    }
}

inline uint32_t Net::p2q(uint32_t i, uint32_t k) {
    uint32_t xbits = p[i].x >> k;                                               // Points are indexed by their x's
    uint32_t ybits = p[i].y >> (m - k);
    return (ybits << (m-k)) | xbits;
}

List Net::netSort() {
    List list(N), tmpList(N);                                                   // We maintain two buffers and swap them
    for (int i = 0; i < N; i++) list[i] = i;                                    // Initialize to a natural order
    for (uint32_t span = N; span > 1; span >>= 1) {                             // Size of sorted sub sets, starting at the whole set
        for (uint32_t slotNo = 0; slotNo < N; slotNo += 2) {                    // Iterate through pairs of slots
            uint32_t relevantBits = slotNo & (span - 1);                        // The set will be permuted in this range only
            uint32_t newSlot0 = slotNo - relevantBits + (relevantBits >> 1);    // Perform a bit rotate over the relevant bits; the least significant being 0;
            uint32_t newSlot1 = newSlot0 + (span >> 1);                         // This is the partial bit rotate for slotNo + 1
            uint32_t toggle = rand() & 1;                                       // Randomly decide which slot maps to which. I tried extracting bits from a single random number, but it exhibited some correlation.
            tmpList[newSlot0] = list[slotNo ^ toggle];
            tmpList[newSlot1] = list[(slotNo + 1) ^ toggle];
        }
        std::swap(list, tmpList);
    }
    return list;
}

inline uint32_t Net::torroidalDistance(uint32_t x1, uint32_t x2) {
    if (x1 < x2) std::swap(x1, x2);
    uint32_t d = x1 - x2;
    if (d >= half) d = N - d;
    return d;
}

void Net::setSigma(double v) {
    sigma = v;
    double sigmaSqx2 = 2 * (v * v * N);
    sigmaSq2Inv = 1.0 / sigmaSqx2;
    fprintf(stderr, "Sigma = %f\n", v);
    for (int x = 0; x < filterRange; x++) {
        kernel[x] = exp(-(x*x) / sigmaSqx2);
    }
}

void Net::setRf(double v) {
    double rf = v * sqrt(N) * dhex;
    rfSq = rf * rf;
}

void Net::setRange(int v) {
    if (v > height / 2 || v <= 0) v = height / 2;
    range = v;
    fprintf(stderr,
        "Optimization neighborhood set to %d neighbor rows/columns\n", v
    );
}

uint32_t Net::getConflictRadiusSq(uint32_t i) {
    uint32_t min(N);
    uint32_t refStratum = p2q(i, main);
    uint32_t Yref = refStratum >> widthBits;
    uint32_t Xref = refStratum & (width - 1);
    for (int dY = -1; dY <= 1; dY++) {
        int Y = (Yref + dY + height) & (height - 1);
        for (int dX = -y2x; dX <= y2x; dX++) {
            if (dX == 0 && dY == 0) continue;
            int X = (Xref + dX + width) & (width - 1);
            int j = q[main][Y * width + X];                                     // Index of neighbor
            uint32_t dx = torroidalDistance(p[j].x, p[i].x);
            uint32_t dy = torroidalDistance(p[j].y, p[i].y);
            uint32_t rr = dx * dx + dy * dy;
            min = std::min(min, rr);
        }
    }
    return min;
}

double Net::conflictRadius() {
    uint32_t min(N);
    for (int i = 0; i < N; i++) {
        min = std::min(min, getConflictRadiusSq(i));
    }
    return conflictRadiusFactor * sqrt(min);
}

int Net::maxConflict(uint32_t i, int k) {
    int stratum = p2q(i, k);
    uint32_t j = q[k][stratum ^ 1];
    int rfSqCurrent = std::min(
        getConflictRadiusSq(i), getConflictRadiusSq(j)
    );
    swap(k, stratum);
    int rfSqNew = std::min(
        getConflictRadiusSq(i), getConflictRadiusSq(j)
    );
    if (rfSqNew > rfSqCurrent) {                                                // If the swap does improve the conflict radius accept it and return 1
        //fprintf(stderr, "[%2d %2d]\n", stratum, k);
        return 1;
    }
    swap(k, stratum);                                                           // Otherwise undo it;
    return 0;                                                                   //  and return 0.
}

int Net::minConflict(uint32_t i, int k) {
    int stratum = p2q(i, k);
    uint32_t j = q[k][stratum ^ 1];
    int rfSqCurrent = std::min(
        getConflictRadiusSq(i), getConflictRadiusSq(j)
    );
    swap(k, stratum);
    int rfSqNew = std::min(
        getConflictRadiusSq(i), getConflictRadiusSq(j)
    );
    if (rfSqNew < rfSqCurrent) {                                                // If the swap worsens the conflict radius accept it and return 1
        fprintf(stderr, "[%2d %2d]\n", stratum, k);
        return 1;
    }
    swap(k, stratum);                                                           // Otherwise undo it;
    return 0;                                                                   //  and return 0.
}

inline double Net::g(int dx) {
    double sum(0.0);
    for (int x = dx; x < filterRange; x += N) {
        sum += kernel[x];
    }
    for (int x = N - dx; x < filterRange; x += N) {
        sum += kernel[x];
    }
    return sum;
}

double Net::cluster(uint32_t i) {
    double sum(0.0);
    uint32_t refStratum = p2q(i, main);
    uint32_t Yref = refStratum >> widthBits;
    uint32_t Xref = refStratum & (width - 1);
    for (int dY = -range; dY < range; dY++) {
        int Y = (Yref + dY + height) & (height - 1);
        for (int dX = -y2x * range; dX < y2x * range; dX++) {
            if (dX == 0 && dY == 0) continue;
            int X = (Xref + dX + width) & (width - 1);
            int j = q[main][Y * width + X];                                     // Index of neighbor
            uint32_t dx = torroidalDistance(p[j].x, p[i].x);
            uint32_t dy = torroidalDistance(p[j].y, p[i].y);
            sum += g(dx) * g(dy);
        }
    }
    return sum;
}


int Net::minCluster(uint32_t i, int k) {
    int stratum = p2q(i, k);
    uint32_t j = q[k][stratum ^ 1];
    double sumCurrent = cluster(i) + cluster(j);
    swap(k, stratum);
    double sumNew = cluster(i) + cluster(j);
    if (sumNew < sumCurrent) {                                                  // If the swap does improve cluster tightness accept it and return 1.
        return 1;
    }
    swap(k, stratum);                                                           // Otherwise undo it;
    return 0;                                                                   //  and return 0.
}

int Net::optimize(int i) {
    int swapCount(0);
    if (useOwen) {
        swapCount += (this->*energyFunction)(i, 0);
        swapCount += (this->*energyFunction)(i, m-1);
        return swapCount;
    }
    for (int k = main; k >= 0; k--) {                                           // Vertical strata, we start in the middle stratifications, which are more significant
        swapCount += (this->*energyFunction)(i, k);
    }
    for (int k = main; k < m; k++) {                                            // horizontal strata, ~
        swapCount += (this->*energyFunction)(i, k);
    }
    return swapCount;
}

void Net::optimize(std::string seq, int iterations) {
    List order(N);
    for (int iteration = 0; iteration < iterations; iteration++) {
        fprintf(stderr, "Iteration %4d:\n", iteration);
        int totalSwapCount(0);
        for (int k = 0; k < seq.length(); k++) {
            switch (seq[k]) {
                case 'f': energyFunction = &Net::maxConflict; break;
                case 'v': energyFunction = &Net::minCluster; break;
                case 'F': energyFunction = &Net::minConflict; break;
                default:
                    fprintf(stderr, "Error: Unknown optimization option.");
                    exit(1);
            }
            shuffle(order);
            int swapCount(0);
            for (int i = 0; i < N && !interuptFlag; i++) {
                swapCount += optimize(order[i]);
            }
            fprintf(
                stderr,
                "  Performed %4d '%c' swaps."
                " Current conflict radius is %0.5f\n",
                swapCount, seq[k], conflictRadius()
            );
            totalSwapCount += swapCount;
        }
        if (interuptFlag || totalSwapCount == 0) break;
    }
}

void Net::printText(String fileName) {                                          // Generate a text printout
    char fullFileName[100];
    std::strcpy(fullFileName, (outputPath + fileName).c_str());
    FILE *file = fopen(fullFileName, "w");
    if (!file) {
        fprintf(stderr, "Failed to open %s\n", fullFileName);
        exit(1);
    }
    fprintf(file, "%d\n", N);
    double res = 1.0 / N;
    for (int stratum = 0; stratum < N; stratum++) {
        int i = q[main][stratum];                                               // We will use main stratification for ordering
        fprintf(file, "%0.9f %0.9f\n", res * p[i].x, res * p[i].y);
    }
    fclose(file);
}

void Net::printEPS(String fileName) {                                           // Generate a text printout
    char fullFileName[100];
    std::strcpy(fullFileName, (outputPath + fileName).c_str());
    FILE *file = fopen(fullFileName, "w");
    if (!file) {
        fprintf(stderr, "Failed to open %s\n", fullFileName);
        exit(1);
    }
    fprintf(
        file,
        "%%!PS-Adobe-3.0 EPSF-3.0\n"
        "%%%%BoundingBox: 0 0 4096 4096\n"
        "/N %d def\n"
        "/unit 1 N div def\n"
        "/depth N log 2 log div round cvi def\n"
        "/r 0.35 unit mul def\n"
        "/linewidth 0.15 unit mul def\n"
        "/p {\n"
        "    2 {linewidth 0.5 mul add exch} repeat\n"
        "    unit linewidth sub dup rectfill\n"
        "} def\n"
        "/plot {\n"
        "    /k exch def\n"
        "    4096 1 linewidth add div dup scale\n"
        "    linewidth 0.5 mul dup translate\n"
        "    2 setlinecap\n"
        "    gsave\n"
        "    0 setlinewidth 1 0.975 0.9375 setrgbcolor\n"
        "    %%0 1 N 1 sub {unit mul 0 moveto 0 1 rlineto} for stroke\n"
        "    %%0 1 N 1 sub {unit mul 0 exch moveto 1 0 rlineto} for stroke\n"
        "    1 0.8 0.5 setrgbcolor\n"
        "    linewidth setlinewidth\n"
        "    0 1 depth k sub bitshift N {\n"
        "        unit mul 0 exch moveto 1 0 rlineto\n"
        "    } for stroke\n"
        "    0 1 k bitshift N {\n"
        "        unit mul 0 moveto 0 1 rlineto\n"
        "    } for stroke\n"
        "    grestore\n"
        "    0.2 setgray\n"
        "    0 1 N 1 sub {\n"
        "        points exch get {unit mul} forall p\n"
        "    } for\n"
        "    showpage\n"
        "} def\n"
        "/points [\n",
        N
    );
    for (int stratum = 0; stratum < N; stratum++) {
        int i = q[main][stratum];                                               // We will use main stratification for ordering
        fprintf(file, "    [%4d %4d]\n", p[i].x, p[i].y);
    }
    fprintf(file, "] def\n0 1 depth {plot} for\n");
    fclose(file);
}

void Net::printFlags(String fileName) {                                         // Print a bitmap of flags
    char fullFileName[100];
    std::strcpy(fullFileName, (outputPath + fileName).c_str());
    FILE *file = fopen(fullFileName, "w");
    if (!file) {
        fprintf(stderr, "Failed to open %s\n", fullFileName);
        exit(1);
    }
    fprintf(file, "P1\n%d %d\n", N / 2, m);                                     // Print output file header
    for (int k = 0; k < m; k++) {
        for (int stratum = 0; stratum < N; stratum += 2) {
            int i = q[k][stratum];
            int j = q[k][stratum + 1];
            char flag = p[i].y < p[j].y ? '0' : '1';
            fprintf(file, "%c ", flag);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

void Net::printComplement(String fileName) {
    char fullFileName[100];
    std::strcpy(fullFileName, (outputPath + fileName).c_str());
    FILE *file = fopen(fullFileName, "w");
    if (!file) {
        fprintf(stderr, "Failed to open %s\n", fullFileName);
        exit(1);
    }
    List xlist(N), tmpList(N);                                                   // We maintain two buffers and swap them
    for (int i = 0; i < N; i++) xlist[i] = i;                                    // Initialize to a natural order
    for (int k = 0; k < m; k++) {
        int span = 1 << (m - k);
        for (uint32_t slotNo = 0; slotNo < N; slotNo += 2) {                    // Iterate through pairs of slots
            uint32_t relevantBits = slotNo & (span - 1);                        // The set will be permuted in this range only
            uint32_t newSlot0 = slotNo - relevantBits + (relevantBits >> 1);    // Perform a bit rotate over the relevant bits; the least significant being 0;
            uint32_t newSlot1 = newSlot0 + (span >> 1);                         // This is the partial bit rotate for slotNo + 1
            uint32_t toggle = p[q[k][slotNo]].y < p[q[k][slotNo + 1]].y ? 1 : 0;// Note that the decision is inverted
            tmpList[newSlot0] = xlist[slotNo ^ toggle];
            tmpList[newSlot1] = xlist[(slotNo + 1) ^ toggle];
        }
        std::swap(xlist, tmpList);
    }

    fprintf(file, "%d\n", N);
    double res = 1.0 / N;
    for (int y = 0; y < N; y++) {
        fprintf(file, "%0.9f %0.9f\n", res * (xlist[y] + 0.5), res * (y + 0.5));
    }
    fclose(file);
}

void Net::upsample() {
    List xlist(N), tmpList(N);                                                  // We maintain two buffers and swap them
    for (int i = 0; i < N; i++) xlist[i] = i;                                   // Initialize to a natural order
    for (int k = 0; k < m; k++) {
        int span = 1 << (m - k);
        for (uint32_t slotNo = 0; slotNo < N; slotNo += 2) {                    // Iterate through pairs of slots
            uint32_t relevantBits = slotNo & (span - 1);                        // The set will be permuted in this range only
            uint32_t newSlot0 = slotNo - relevantBits + (relevantBits >> 1);    // Perform a bit rotate over the relevant bits; the least significant being 0;
            uint32_t newSlot1 = newSlot0 + (span >> 1);                         // This is the partial bit rotate for slotNo + 1
            uint32_t toggle = p[q[k][slotNo]].y < p[q[k][slotNo + 1]].y ? 1 : 0;// Note that the decision is inverted
            tmpList[newSlot0] = xlist[slotNo ^ toggle];
            tmpList[newSlot1] = xlist[(slotNo + 1) ^ toggle];
        }
        std::swap(xlist, tmpList);
    }
    p.resize(2 * N);
    for (int i = 0; i < N; i++) {
        p[    i] = {2 * p[i].x, 2 * p[i].y};
        p[N + i] = {2 * xlist[i] + 1, 2 * i + 1};
    }
    m++;
    N *= 2;
    init();
    useOwen = true;
}


static void signalHandler(int signalCode) {
    fprintf(stderr, "Aborting ...\n");
    interuptFlag = true;
}

const char *USAGE_MESSAGE = "Usage: %s [options] <point count>\n"
"Options:\n"
"    -l <flagsFileName>\n"
"    -n <number of iterations>\n"
"    -q <optimization letter sequence>\n"
"    -s <sigma for Gaussian filtering>\n"
"    -r <target conflict radius>\n"
"    -b <log2optimizationBlockSize>\n"
"    -R <Neighborhood size for cluster optimization>\n"
"Optimization sequences:\n"
"    f: conflict;\n"
"    F: worsens conflict;\n"
"    v: void-n-cluster;\n"
"    V: void-n-cluster with Poisson disk enhancement\n"
"    e: electrostatic\n"
"    j: 1D conflict\n"
"    c: Poisson Disk\n";

int main(int argc,char **argv) {
    srand(time(NULL));
    srand48(time(NULL));
    int opt;                                                                    // For use by getopt, the command line options utility
    int iterations = 0;
    double sigma = 0.4;
    char *flagsFileName = NULL;
    std::string optimizationSequence = "f";
    double rf = 0.75;
    uint32_t bBits = 0;
    int range = 0;
    fprintf(
        stderr,
        "Generate a point set of size N, where N is a power of 2\n"
    );
    while ((opt = getopt(argc, argv, "l:n:q:s:r:b:R:")) != -1) {                        // Modify default settings with command line options
        switch (opt) {
            case 'l': flagsFileName = optarg; break;
            case 'n': iterations = atoi(optarg); break;
            case 'q': optimizationSequence = optarg; break;
            case 's': sigma = atof(optarg); break;
            case 'r': rf = atof(optarg); break;
            case 'b': bBits = atoi(optarg); break;
            case 'R': range = atoi(optarg); break;
            default: fprintf(stderr, USAGE_MESSAGE, argv[0]); exit(1);
        }
    }
    if (optind > argc - 1) {
        fprintf(stderr, USAGE_MESSAGE, argv[0]); exit(1);
    }
    signal(SIGINT, signalHandler);
    uint32_t pointCount = atoi(argv[optind]);
    Net *net;
    if (flagsFileName != NULL) {
        net = new Net(flagsFileName);
    }
    else {
        net = new Net(pointCount);
    }
    net->setSigma(sigma);
    net->setRf(rf);
    net->setRange(range);
    net->printEPS("initial.eps");
    fprintf(stderr, "Optimizing ..\n");
    clock_t t0 = clock();
    if (iterations) net->optimize(optimizationSequence, iterations);
    clock_t t1 = clock();
    double totalTime = (double)(t1 - t0) / CLOCKS_PER_SEC;
    fprintf(stderr, "\ndone! Total time = %.6fs\n", totalTime);
    net->printText();
    net->printComplement();
    net->printEPS();
    net->printFlags();
//     net->upsample();
//     net->optimize(optimizationSequence, iterations);
//     net->printText("union.txt");
    fprintf(stderr, "rf = %0.5f\n", net->conflictRadius());
}





#include <iostream>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl/segmentation/approximate_progressive_morphological_filter.h>

#include "Timer.hpp"
typedef pcl::PointXYZ POINT;
using namespace std;

float idist, Maxdist, Slope = 1.0;
int MaxWindowSize = 15;

int apmf(pcl::PointCloud<POINT>::Ptr cloud, pcl::PointCloud<POINT>::Ptr cloud_o, pcl::PointCloud<POINT>::Ptr cloud_g, bool para = true, bool save = false)
{
    static size_t cnt = 0;
    pcl::PointIndicesPtr ground(new pcl::PointIndices);
    // std::cerr << "Cloud before filtering: " << cloud->points.size() << std::endl;
    // auto startTime = std::chrono::steady_clock::now();
    {
        if (para)
        {
            pcl::ApproximateProgressiveMorphologicalFilter<POINT> pmf;
            pmf.setInputCloud(cloud);
            pmf.setMaxWindowSize(MaxWindowSize); // set window size
            pmf.setSlope(Slope);                 // slope
            pmf.setInitialDistance(idist);
            pmf.setMaxDistance(Maxdist);
            // Timer t("apmf");
            pmf.extract(ground->indices);
            // printf("para\n");
        }
        else
        {
            pcl::ProgressiveMorphologicalFilter<POINT> pmf;
            pmf.setInputCloud(cloud);
            pmf.setMaxWindowSize(MaxWindowSize); // set window size
            pmf.setSlope(Slope);                 // slope
            pmf.setInitialDistance(idist);
            pmf.setMaxDistance(Maxdist);
            // Timer t("pmf");
            pmf.extract(ground->indices);
            // printf("no para\n");
        }
    }

    if (save)
    {
        pcl::ExtractIndices<POINT> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(ground);
        extract.filter(*cloud_o);

        std::string pcdFileName = "../pcd/" + to_string(cnt) + ".pcd";
        if (cloud_o->points.size())
            pcl::io::savePCDFile<POINT>(pcdFileName, *cloud_o, false);

        // Extract non-ground returns
        extract.setNegative(true);
        extract.filter(*cloud_g);

        // auto endTime = std::chrono::steady_clock::now();
        // auto ellapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        // std::cout << "Ellapse-Time: " << ellaps  edTime.count() << " milliseconds." << std::endl;
        // std::cerr << "Object cloud after filtering: " << cloud_g->points.size() << std::endl;

        pcdFileName = "../pcd/" + to_string(cnt) + "o.pcd";
        if (cloud_g->points.size())
            pcl::io::savePCDFile<POINT>(pcdFileName, *cloud_g, false);
        cnt++;
    }

    return 0;
}
int main2(int argc, char **av)
{
    if (argc < 2)
    {
        printf("./pmf ./out.pcd 0.5 3.0 15 0.1 \n idist=%f Maxdist= %f\n", idist, Maxdist);
        return -1;
    }
    pcl::PointCloud<POINT>::Ptr cloud(new pcl::PointCloud<POINT>);
    pcl::PointCloud<POINT>::Ptr cloudo(new pcl::PointCloud<POINT>);
    pcl::PointCloud<POINT>::Ptr cloudg(new pcl::PointCloud<POINT>);

    string filename(av[1]);
    idist = atof(av[2]), Maxdist = atof(av[3]), MaxWindowSize = atoi(av[4]), Slope = atof(av[5]);

    vector<pcl::PointCloud<POINT>::Ptr> pts;
    for (int i = 0; i < 100; i++)
    {
        pcl::PointCloud<POINT>::Ptr cloud(new pcl::PointCloud<POINT>);
        pcl::io::loadPCDFile(filename, *cloud);
        pts.push_back(cloud);
    }

    apmf(cloud, cloudo, cloudg, atoi(av[6]));
}
int main(int argc, char **av)
{
    if (argc < 2)
    {
        printf("./pmf ./out.pcd 0.5 3.0 15 0.1 \n idist=%f Maxdist= %f\n", idist, Maxdist);
        return -1;
    }
    pcl::PointCloud<POINT>::Ptr cloud(new pcl::PointCloud<POINT>);

    string filename(av[1]);
    idist = atof(av[2]), Maxdist = atof(av[3]), MaxWindowSize = atoi(av[4]), Slope = atof(av[5]);
    char names[256];
    vector<pcl::PointCloud<POINT>::Ptr> pts;
    for (int i = 0; i < 100; i++)
    {
        pcl::PointCloud<POINT>::Ptr cloud(new pcl::PointCloud<POINT>);
        sprintf(names, "/home/u20/lei/home/lei/桌面/00/bin_to_pcd/build/%d.pcd", i);
        pcl::io::loadPCDFile(string(names), *cloud);
        pts.push_back(cloud);
    }

    {
        Timer t("apmf all");
        for (int i = 0; i < 100; i++)
        {
            pcl::PointCloud<POINT>::Ptr cloudo(new pcl::PointCloud<POINT>);
            pcl::PointCloud<POINT>::Ptr cloudg(new pcl::PointCloud<POINT>);
            apmf(pts[i], cloudo, cloudg, atoi(av[6]));
        }
    }

    // apmf(cloud, cloudo, cloudg, atoi(av[6]));
}
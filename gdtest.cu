#include <iostream>
#include <fstream>
#include "cuVector.cuh"
// #include <pcl/io/pcd_io.h>
#include "pcd_io.h"
#include <pcl/point_types.h>
#include "Timer.hpp"
#include "apmf.h"
#include <pcl/filters/filter.h>
using namespace std;

typedef pcl::PointXYZI MPOINT;

template <typename T>
void ReadPCD_XYZL(std::vector<T> &vps, string filename)
{
  // Timer mt("read");
  FILE *fp;
  fp = fopen(filename.c_str(), "r");
  char str[1024];
  size_t num = 0;
  for (int i = 0; i < 11; i++)
  {
    int ret = fscanf(fp, "%s %[^\n]\n", str + 512, str);
    if (strcmp("POINTS", str + 512) == 0)
    {
      num = atoi(str);
    }
  }
  // cout << num << endl;
  vps.resize(num);
  int la;
  for (size_t i = 0; i < num; i++)
  {
    auto &p = vps[i];
    int ret = fscanf(fp, "%f %f %f %f\n", &p.x, &p.y, &p.z, &p.intensity);
  }
  fclose(fp);
}

template <typename POINT>
void pcdwrite(std::string file_name, std::vector<POINT> &cloud, std::vector<int> &index)
{
  // Timer mt("write");

  std::bitset<1024 * 1024 * 8 * 4> foo;
  {
    // Timer mt("save");
    for (int i = 0; i < index.size(); i++)
    {
      foo.set(index[i]);
    }
  }
  FILE *fp = fopen(file_name.c_str(), "w");
  fprintf(fp, "# .PCD v0.7 - Point Cloud Data file format\n");
  fprintf(fp, "VERSION 0.7\n");
  fprintf(fp, "FIELDS x y z intensity label\n");
  fprintf(fp, "SIZE 4 4 4 4 2\n");
  fprintf(fp, "TYPE F F F F U\n");
  fprintf(fp, "COUNT 1 1 1 1 1\n");
  fprintf(fp, "WIDTH %d\n", 124669);
  fprintf(fp, "HEIGHT %d\n", 1);
  fprintf(fp, "VIEWPOINT 0 0 0 1 0 0 0\n");
  fprintf(fp, "POINTS %ld\n", cloud.size());
  fprintf(fp, "DATA ascii\n");
  int cnt = 0;
  for (int i = 0; i < cloud.size(); i++)
  {
    auto &it = cloud[i];
    if (foo.test(i))
    {
      fprintf(fp, "%f %f %f %d %d\n", it.x, it.y, it.z, (int)it.intensity, 1);
      // printf("%f %f %f\n", it.x, it.y, it.z);
      cnt++;
    }
    else
    {
      fprintf(fp, "%f %f %f %d %d\n", it.x, it.y, it.z, (int)it.intensity, 0);
    }
  }
  // cout << file_name << " " << cnt << endl;
  fclose(fp);
}

void xyzwritePCD(std::string file_name, pcl::PointCloud<pcl::PointXYZ> &cloud, std::vector<int> &index)
{

  pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_xyzi(new pcl::PointCloud<pcl::PointXYZL>());

  pcl::copyPointCloud(cloud, *cloud_xyzi);

  std::bitset<1024 * 1024 * 16> foo;
  for (int i = 0; i < index.size(); i++)
  {
    foo.set(index[i]);
  }
  for (int i = 0; i < cloud.points.size(); i++)
  {
    cloud_xyzi->points[i].label = foo.test(i);
    if (cloud_xyzi->points[i].label)
    {
      printf("%f %f %f\n", cloud_xyzi->points[i].x, cloud_xyzi->points[i].y, cloud_xyzi->points[i].z);
    }
  }
}
float idist, Maxdist, Slope = 1.0;
int MaxWindowSize = 15;

void cudaPMF2(string filename)
{
  std::vector<int> indices;
  std::vector<MPOINT> __mvs;
  cout << filename << endl;
  ReadPCD_XYZL(__mvs, filename);
  {
    Timer mt("cuda all");

    for (int i = 0; i < 30; i++)
    {
      indices.resize(0);
      pcl::Apmf<MPOINT> pmf;
      pmf.setMaxWindowSize(MaxWindowSize);
      pmf.setSlope(Slope);
      pmf.setInitialDistance(idist); // 0.5
      pmf.setMaxDistance(Maxdist);   // 3
      pmf.max_height_ = 3;
      pmf.extract(indices, __mvs);
    }
  }

  pcdwrite("out.pcd", __mvs, indices);
}
int main2(int argc, char **av)
{
  if (argc < 4)
  {
    printf("./gdtest ./out.pcd 0.5 3.0 15 0.1 \n idist=%f Maxdist= %f\n", idist, Maxdist);
    return -1;
  }
  string filename(av[1]);
  idist = atof(av[2]), Maxdist = atof(av[3]), MaxWindowSize = atoi(av[4]), Slope = atof(av[5]);
  cudaPMF2("../data/a.pcd");
}

void cudaPMF(std::vector<MPOINT> __mvs)
{
  std::vector<int> indices;

  indices.resize(0);
  pcl::Apmf<MPOINT> pmf;
  pmf.setMaxWindowSize(MaxWindowSize);
  pmf.setSlope(Slope);
  pmf.setInitialDistance(idist); // 0.5
  pmf.setMaxDistance(Maxdist);   // 3
  pmf.max_height_ = 3;
  pmf.extract(indices, __mvs);
}

int main(int argc, char **av)
{
  if (argc < 4)
  {
    printf("./gdtest ./out.pcd 0.5 3.0 15 0.1 \n idist=%f Maxdist= %f\n", idist, Maxdist);
    return -1;
  }
  string filename(av[1]);
  idist = atof(av[2]), Maxdist = atof(av[3]), MaxWindowSize = atoi(av[4]), Slope = atof(av[5]);
  char names[256];
  vector<std::vector<MPOINT> *> pts;
  for (int i = 0; i < 100; i++)
  {
    std::vector<MPOINT> *pmvs = new std::vector<MPOINT>();
    sprintf(names, "/home/u20/lei/home/lei/桌面/00/bin_to_pcd/build/%d.pcd", i);
    ReadPCD_XYZL(*pmvs, names);
    pts.push_back(pmvs);
  }
  {
    Timer mt("cuda all");
    for (int i = 0; i < 100; i++)
    {
      cudaPMF(*pts[i]);
    }
  }
}

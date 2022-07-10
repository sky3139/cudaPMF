#include <iostream>
#include <fstream>
#include "cuVector.cuh"
// #include <pcl/io/pcd_io.h>
#include "pcd_io.h"
#include <pcl/point_types.h>
#include "Timer.hpp"

#include "apmf.h"
#include <pcl/filters/filter.h>
#include "threadpool.h"
using namespace std;

typedef pcl::PointXYZI MPOINT;

template <typename T>
void ReadPCD_XYZL(std::vector<T> &vps, string filename)
{
  // Timer mt("read");
  FILE *fp;
  fp = fopen(filename.c_str(), "r");
  char str[1024];
  for (int i = 0; i < 11; i++)
  {
    int ret = fscanf(fp, "%[^\n]\n", str);
  }
  vps.resize(230400);
  for (size_t i = 0; i < 230400; i++)
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
  fprintf(fp, "WIDTH %d\n", 1800);
  fprintf(fp, "HEIGHT %d\n", 128);
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

void cudaPMF2(string path, string filename)
{
  std::vector<int> indices;
  std::vector<MPOINT> __mvs;
  // cout << path + filename << endl;
  ReadPCD_XYZL(__mvs, path + filename);
  pcl::Apmf<MPOINT> pmf;
  pmf.setMaxWindowSize(35);
  pmf.setSlope(1.0f);
  pmf.setInitialDistance(0.4f); // 0.5
  pmf.setMaxDistance(3.0f);     // 3
  pmf.max_height_ = 3;
  pmf.extract(indices, __mvs);
  pcdwrite("/task6_result/spirt/" + filename, __mvs, indices);
}
int main(int c,char **)
{
  Timer mt("write");
  string inPath("/task6/");
  DIR *dir = opendir(inPath.c_str());
  struct dirent *ptr;

  std::vector<string> files;
  while ((ptr = readdir(dir)) != NULL)
  {
    if (ptr->d_name[0] == '.')
      continue;
    files.push_back(string(ptr->d_name));
  }
  closedir(dir);
  delete ptr;
  threadpool executor(12);
  for (int i = 0; i < files.size(); i++) // files.size()
  {
    executor.commit(cudaPMF2, inPath, files[i]);
  }
  executor.waitTask();
}

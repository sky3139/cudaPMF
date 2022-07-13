#include <iostream>
#include <fstream>
#include "cuVector.cuh"
// #include <pcl/io/pcd_io.h>
// #include "pcd_io.h"
#include <pcl/point_types.h>
#include "Timer.hpp"
#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>

#include "apmf.h"
// #include <pcl/filters/filter.h>
#include "threadpool.h"
#include "rw.h"
using namespace std;

// struct MPOINT
// {
//   float x, y, z;
//   uint32_t label;

//   Eigen::Array4f getArray4fMap()
//   {
//     Eigen::Array4f ret;
//     ret.x() = x, ret.y()  = y, ret.z () = z, ret.w()  = 0;
//     return ret;
//   }
// };

typedef pcl::PointXYZI MPOINT;

template <typename T>
void readPLY(std::vector<T> &vps, string filename)
{

  ifstream fin(filename, ios_base::in); //開檔
  if (!fin.is_open())
  {
    cout << "Cannot read the file." << endl;
    exit(0);
  }
  string str;
  int vertex, face;
  char ch;
  bool ascii = true;
  while (!fin.eof())
  {
    fin.get(ch);
    if (ch != ' ' && ch != '\t' && ch != '\n')
    {
      str.push_back(ch);
    }
    else
    {
      //取得vertex個數
      if (str == "POINTS" || str == "vertex")
      {
        getline(fin, str, '\n');
        vertex = atoi(str.c_str());
      }
      else if (str == "DATA")
      {
        getline(fin, str, '\n');
        break;
      }
      else if (str == "end_header")
      {
        break;
      }
      else if (str == "binary_little_endian")
      {
        ascii = false;
      }
      str.clear();
    }
  }
  int recv;
  if (ascii)
  {
    while (1)
    {
      T a;
      fin >> a.x >> a.y >> a.z >> recv >> recv >> recv;
      if (!fin.good())
        break;
      a.z = -a.z;
      vps.push_back(a);
    }
  }
  else
  {
    float recv[10];
    while (1)
    {
      T a;
      fin.read((char *)&a.x, 3 * sizeof(float));

      fin.read((char *)recv, 9 * sizeof(float));
      // if (a.z < -0.5)
      //   continue;
      if (!fin.good())
        break;
      vps.push_back(a);
    }
  }
  // cout << a.x << " " << a.y << a.z << endl;
  // vps.resize(vertex);
  //
  cout << vertex << " " << vps.size() << " " << ascii << endl;
}

template <typename POINT>
void pcdwrite(std::string file_name, std::vector<POINT> &cloud, std::vector<int> &index, bool ascii = true)
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
  fprintf(fp, "FIELDS x y z label\n");
  fprintf(fp, "SIZE 4 4 4 2\n");
  fprintf(fp, "TYPE F F F U\n");
  fprintf(fp, "COUNT 1 1 1 1\n");
  fprintf(fp, "WIDTH %ld\n", cloud.size());
  fprintf(fp, "HEIGHT %d\n", 1);
  fprintf(fp, "VIEWPOINT 0 0 0 1 0 0 0\n");
  fprintf(fp, "POINTS %ld\n", cloud.size());
  if (ascii)
    fprintf(fp, "DATA ascii\n");
  else
    fprintf(fp, "DATA binary\n");
  int cnt = 0;
  if (ascii)
    for (int i = 0; i < cloud.size(); i++)
    {
      auto &it = cloud[i];
      if (foo.test(i))
      {
        fprintf(fp, "%f %f %f %d\n", it.x, it.y, it.z, 1);
        // printf("%f %f %f\n", it.x, it.y, it.z);
        cnt++;
      }
      else
      {
        fprintf(fp, "%f %f %f %d\n", it.x, it.y, it.z, 0);
      }
    }
  else
  {
    for (int i = 0; i < cloud.size(); i++)
    {
      auto &it = cloud[i];
      uint16_t ret = foo.test(i);
      fwrite(&it.x, sizeof(float), 3, fp);
      fwrite(&ret, sizeof(uint16_t), 1, fp);
    }
  }
  // cout << file_name << " " << cnt << endl;
  fclose(fp);
}

void cudaPMF2(string path, string filename)
{
  // Timer t("all");
  std::vector<int> indices;
  std::vector<MPOINT> __mvs;
  // cout << path + filename << endl;
  readPLY(__mvs, path + filename);
  pcl::Apmf<MPOINT> pmf;
  pmf.setMaxWindowSize(35);
  pmf.setSlope(1.0f);
  pmf.setInitialDistance(0.4f); // 0.5
  pmf.setMaxDistance(3.0f);     // 3
  pmf.max_height_ = 3;
  pmf.extract(indices, __mvs);
  filename = filename.substr(0, filename.size() - 4);
  pcdwrite("pcd/" + filename + ".pcd", __mvs, indices, false);
}

int main(int, char **)
{
  Timer mt("all");
  string inPath("/home/u20/d2/cudaPMF/build/input/");
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
    cout << inPath << files[i] << endl;
    executor.commit(cudaPMF2, inPath, files[i]);
    break;
  }
  executor.waitTask();
}

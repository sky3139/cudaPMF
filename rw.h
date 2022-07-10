#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
using namespace std;

#pragma pack(push, 1)
struct POINTL
{
    float x, y, z;
    uint32_t label;
};
void writepcd(std::string str, void *pvps, size_t size);
template <typename T>
void ReadPCD_XYZL(std::vector<T> &vps, std::string filename);
template <typename T>
void readPLY(std::vector<T> &vps, string filename);

#pragma pack(pop)
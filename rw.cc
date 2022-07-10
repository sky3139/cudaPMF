

#include "rw.h"
#include <vector>

using namespace std;
void writepcd(std::string str, void *pvps, size_t size)
{
    fstream fs(str, std::ios::out);
    fs << "# .PCD v0.7 - Point Cloud Data file format"
       << "VERSION 0.7\n"
       << "FIELDS x y z label\n"
       << "SIZE 4 4 4 4\n"
       << "TYPE F F F U\n"
       << "COUNT 1 1 1 1\n"
       << "WIDTH " << size
       << "\nHEIGHT 1\n"
       << "VIEWPOINT 0 0 0 1 0 0 0\n"
       << "POINTS " << size
       << "\nDATA binary\n"; //  ascii

    fs.write((char *)pvps, sizeof(POINTL) * size);

    fs.close();
}
template <typename T>
void ReadPCD_XYZL(std::vector<T> &vps, string filename)
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
            if (str == "POINTS")
            {
                str.clear();
                getline(fin, str, '\n');
                vertex = atoi(str.c_str());
            }
            //取得face個數
            else if (str == "face")
            {
                str.clear();
                getline(fin, str, '\n');
                face = atoi(str.c_str());
            }
            else if (str == "DATA")
            {
                str.clear();
                getline(fin, str, '\n');
                break;
            }
            else
                str.clear();
        }
    }
    vps.resize(vertex);
    fin.read((char *)vps.data(), vertex * sizeof(T));
    cout << vps.size() << endl;
}
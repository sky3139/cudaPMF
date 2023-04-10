// InputReader.cpp: implementation of the InputReader class.
//
//////////////////////////////////////////////////////////////////////

#include "InputReader.h"
#include <fstream>
#include <sstream>
using namespace std;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

void InputReader::Read(string filename) throw()
{
	ifstream in(filename.c_str());
	if (!in)
	{
		assert(0);
		return;
	}

	float x, y, z, r, g, b;
	m_vecPoints.clear();
	while (!in.eof())
	{
		string line;
		getline(in, line);
		istringstream iss(line);
		iss >> x >> y >> z >> r >> g >> b;
		// int i = (int)(0.299 * posClr.R + 0.587 * posClr.G + 0.114 * posClr.B);
		//  Gray = R*0.299 + G*0.587 + B*0.114
		// z=r/255.0f;
		m_vecPoints.push_back(Point3D(x, y, z));
	}
}
void InputReader::ReadPLY(string filename)
{
	ifstream fin(filename, ios_base::in); //開檔

	if (!fin.is_open())
	{
		cout << "Cannot read the file." << endl;
		cout << "Please check again." << endl;
		exit(0);
	}

	string str;
	int vertex, face;
	char ch;
	/*讀取header*/

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
			if (str == "vertex")
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
			else if (str == "end_header")
			{
				str.clear();
				break;
			}
			else
				str.clear();
		}
	}
	float x, y, z, r, g, b;
	m_vecPoints.clear();
	while (!fin.eof())
	{
		fin >> x >> z >> y >> r >> g >> b;
		// int i = (int)(0.299 * posClr.R + 0.587 * posClr.G + 0.114 * posClr.B);
		//  Gray = R*0.299 + G*0.587 + B*0.114
		// z=r/255.0f;

		m_vecPoints.push_back(Point3D(x, y, z));
	}
	m_vecPoints.pop_back();
	movecenter();
	cout << m_vecPoints.size() << endl;
}

void InputReader::Dump(string str = "src.xyz")
{

	fstream fs(str, std::ios::out);
	for (int i = 0; i < m_vecPoints.size(); i++)
		fs << m_vecPoints[i].x << "\t" << m_vecPoints[i].y << "\t" << m_vecPoints[i].z << endl;
	fs.close();
}
void InputReader::movecenter()
{
	Point3D psum;
	for (auto &p : m_vecPoints)
	{
		psum.x += p.x;
		psum.y += p.y;
		psum.z += p.z;
	}
	Point3D avg(psum.x / m_vecPoints.size(), psum.y / m_vecPoints.size(),psum.z / m_vecPoints.size());
	for (auto &p : m_vecPoints)
	{
		p.x -= avg.x;
		p.y -= avg.y;
		p.z -= avg.z;
	}
}
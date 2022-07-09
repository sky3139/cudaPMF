#pragma once
template<typename T>
struct TPoint2D
{
	TPoint2D() : x(0), y(0) {}
	TPoint2D(T xx, T yy) : x(xx), y(yy) {}
	T x;
	T y;
};

typedef TPoint2D<double> Point2D;

template<typename T>
struct TPoint3D
{
	TPoint3D() : x(0), y(0), z(0) {}
	TPoint3D(T xx, T yy, T zz) : x(xx), y(yy), z(zz) {}
	T x;
	T y;
	T z;
};

typedef TPoint3D<float> Point3D;
#include <iostream>
#include <string>
#include <vector>
#include <cassert>
using namespace std;

class InputReader
{
public:
	InputReader() throw() {}
	InputReader(string filename) throw() { Read(filename); }
	void ReadPLY(string filename);

	void Read(string filename) throw();
	const vector<Point3D> &Get3DPoints() const { return m_vecPoints; }
	void Dump(string st);
	void movecenter();
	void print()
	{
		for (int i = 0; i < m_vecPoints.size(); i++)
		{
			// cout << i << " " << m_vecPoints[i].x << " " << m_vecPoints[i].y << " " << m_vecPoints[i].z << endl;
			cout << m_vecPoints[i].x << " " << m_vecPoints[i].y << " " << m_vecPoints[i].z << endl;
		}
	}

private:
	vector<Point3D> m_vecPoints;
};

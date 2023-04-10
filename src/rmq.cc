#include <cstdio>
#include <cstring>
#include <algorithm>
using namespace std;
const int MAXN = 400;
int val[MAXN][MAXN];
int dmin[MAXN][MAXN][32][32];
int dmax[MAXN][MAXN][32][32];
class RMQ
{
public:
    void initRMQ(int n, int m)
    {
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++)
                dmin[i][j][0][0] = dmax[i][j][0][0] = val[i][j];
        for (int ii = 0; (1 << ii) <= n; ii++)
            for (int jj = 0; (1 << jj) <= m; jj++)
                if (ii + jj)
                    for (int i = 1; i + (1 << ii) - 1 <= n; i++)
                        for (int j = 1; j + (1 << jj) - 1 <= m; j++)
                            if (ii)
                            {
                                dmin[i][j][ii][jj] = min(dmin[i][j][ii - 1][jj], dmin[i + (1 << (ii - 1))][j][ii - 1][jj]);
                                dmax[i][j][ii][jj] = max(dmax[i][j][ii - 1][jj], dmax[i + (1 << (ii - 1))][j][ii - 1][jj]);
                            }
                            else
                            {
                                dmin[i][j][ii][jj] = min(dmin[i][j][ii][jj - 1], dmin[i][j + (1 << (jj - 1))][ii][jj - 1]);
                                dmax[i][j][ii][jj] = max(dmax[i][j][ii][jj - 1], dmax[i][j + (1 << (jj - 1))][ii][jj - 1]);
                            }
    }
    int getMax(int x1, int y1, int x2, int y2)
    {
        int k1 = 0;
        while ((1 << (k1 + 1)) <= x2 - x1 + 1)
            k1++;
        int k2 = 0;
        while ((1 << (k2 + 1)) <= y2 - y1 + 1)
            k2++;
        x2 = x2 - (1 << k1) + 1;
        y2 = y2 - (1 << k2) + 1;
        return max(max(dmax[x1][y1][k1][k2], dmax[x1][y2][k1][k2]), max(dmax[x2][y1][k1][k2], dmax[x2][y2][k1][k2]));
    }
    int getMin(int x1, int y1, int x2, int y2)
    {
        int k1 = 0;
        while ((1 << (k1 + 1)) <= x2 - x1 + 1)
            k1++;
        int k2 = 0;
        while ((1 << (k2 + 1)) <= y2 - y1 + 1)
            k2++;
        x2 = x2 - (1 << k1) + 1;
        y2 = y2 - (1 << k2) + 1;
        return min(min(dmin[x1][y1][k1][k2], dmin[x1][y2][k1][k2]), min(dmin[x2][y1][k1][k2], dmin[x2][y2][k1][k2]));
    }
};

int main()
{
    int n = 5, b = 3;
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++)
            scanf("%d", &val[i][j]);
    RMQ rmq;
    rmq.initRMQ(n, n);

    int x1 = 1, y1 = 2;
    printf("%d %d\n", rmq.getMax(x1, y1, x1 + b - 1, y1 + b - 1), rmq.getMin(x1, y1, x1 + b - 1, y1 + b - 1));
    return 0;
}

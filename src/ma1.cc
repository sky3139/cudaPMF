#include <bits/stdc++.h>

using namespace std;

const int MAXN = 1e3 + 13;
int n, m, x, y, L,  a[MAXN][MAXN];
struct tree
{
    int x1, y1, x2, y2;
    int Max, Min;
} tr[MAXN * MAXN * 4];

int son(int p, int x) { return p * 4 - 2 + x; }

void pushup1(int rt)
{
    tr[rt].Max = max(tr[son(rt, 0)].Max, tr[son(rt, 1)].Max);
    tr[rt].Min = min(tr[son(rt, 0)].Min, tr[son(rt, 1)].Min);
    for (int i = 2; i < 4; ++i)
    {
        tr[rt].Max = max(tr[rt].Max, tr[son(rt, i)].Max);
        tr[rt].Min = min(tr[rt].Min, tr[son(rt, i)].Min);
    }
}

void pushup2(int rt)
{ //二叉树的pushup
    tr[rt].Max = max(tr[son(rt, 0)].Max, tr[son(rt, 1)].Max);
    tr[rt].Min = min(tr[son(rt, 0)].Min, tr[son(rt, 1)].Min);
}

void build(int rt, int X1, int Y1, int X2, int Y2)
{
    tr[rt].x1 = X1, tr[rt].x2 = X2;
    tr[rt].y1 = Y1, tr[rt].y2 = Y2;
    if (X1 == X2 && Y1 == Y2)
    {
        tr[rt].Max = a[X1][Y1];
        tr[rt].Min = a[X1][Y1];
        return;
    }
    int midx = (X1 + X2) >> 1;
    int midy = (Y1 + Y2) >> 1;
    if (X1 == X2)
    {
        build(son(rt, 0), X1, Y1, X2, midy);
        build(son(rt, 1), X1, midy + 1, X2, Y2);
        pushup2(rt);
    }
    else if (Y1 == Y2)
    {
        build(son(rt, 0), X1, Y1, midx, Y2);
        build(son(rt, 1), midx + 1, Y1, X2, Y2);
        pushup2(rt);
    }
    else
    {
        build(son(rt, 0), X1, Y1, midx, midy);
        build(son(rt, 1), midx + 1, Y1, X2, midy);
        build(son(rt, 2), X1, midy + 1, midx, Y2);
        build(son(rt, 3), midx + 1, midy + 1, X2, Y2);
        pushup1(rt);
    }
}

int A, B;
void query(int rt, int X1, int Y1, int X2, int Y2)
{
    if (tr[rt].x1 > X2 || tr[rt].x2 < X1 || tr[rt].y1 > Y2 || tr[rt].y2 < Y1)
        return;
    if (tr[rt].x1 >= X1 && tr[rt].x2 <= X2 && tr[rt].y1 >= Y1 && tr[rt].y2 <= Y2)
    {
        A = max(A, tr[rt].Max);
        B = min(B, tr[rt].Min);
        return;
    }
    if (tr[rt].x1 == tr[rt].x2 || tr[rt].y1 == tr[rt].y2)
    {
        query(son(rt, 0), X1, Y1, X2, Y2);
        query(son(rt, 1), X1, Y1, X2, Y2);
    }
    else
    {
        query(son(rt, 0), X1, Y1, X2, Y2);
        query(son(rt, 1), X1, Y1, X2, Y2);
        query(son(rt, 2), X1, Y1, X2, Y2);
        query(son(rt, 3), X1, Y1, X2, Y2);
    }
}

void modify(int rt, int X1, int Y1, int X2, int Y2, int val)
{
    if (tr[rt].x1 > X2 || tr[rt].x2 < X1 || tr[rt].y1 > Y2 || tr[rt].y2 < Y1)
        return;
    if (tr[rt].x1 >= X1 && tr[rt].x2 <= X2 && tr[rt].y1 >= Y1 && tr[rt].y2 <= Y2)
    {
        tr[rt].Max = val;
        tr[rt].Min = val;
        return;
    }
    if (tr[rt].x1 == tr[rt].x2 || tr[rt].y1 == tr[rt].y2)
    {
        modify(son(rt, 0), X1, Y1, X2, Y2, val);
        modify(son(rt, 1), X1, Y1, X2, Y2, val);
        pushup2(rt);
    }
    else
    {
        modify(son(rt, 0), X1, Y1, X2, Y2, val);
        modify(son(rt, 1), X1, Y1, X2, Y2, val);
        modify(son(rt, 2), X1, Y1, X2, Y2, val);
        modify(son(rt, 3), X1, Y1, X2, Y2, val);
        pushup1(rt);
    }
}

int main()
{
    n=5;
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= n; ++j)
            scanf("%d", &a[i][j]);
    build(1, 1, 1, n, n); // 建树
    int X1, X2, Y1, Y2;

    scanf("%d %d %d", &x, &y, &L);
    L = (L - 1) >> 1;
    X1 = max(1, x - L), Y1 = max(1, y - L); // 找到边界
    X2 = min(n, x + L), Y2 = min(n, y + L);
    A = 0, B = 1e9 + 19;
    query(1, X1, Y1, X2, Y2);            //区间查询
    modify(1, x, y, x, y, (A + B) >> 1); // 单点修改
    printf("%d %d\n", A,B);
    return 0;
}
// // 5
// 5 1 2 6 3
// 1 3 5 2 7
// 7 2 4 6 1
// 9 9 8 6 5
// 0 6 9 3 9

// 5 3 1
// 5 1 2 6 3
// 1 3 5 2 7
// 7 2 4 6 1
// 9 9 8 6 5
// 0 6 9 3 9
// 1 2
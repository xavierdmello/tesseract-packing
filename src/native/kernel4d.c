// Native kernel for the 4D packing search: penalty, analytic gradient and Adam steps.
// Mirrors src/kernel.py (Batch.violation + torch.optim.Adam) for D = 4.
//
// Per configuration b (of B), with n cubes and P = n(n-1)/2 pairs (i<j, row-major like torch.triu_indices):
//   c[b][i][4]    centres
//   p[b][i][8]    quaternion pair (a, q) -> R = L(a/|a|) * R(q/|q|), columns = cube axes
//   W[b][k][4]    unnormalised separating normal of pair k;  Bh[b][k] its offset
//   s[b], r[b]    container side, rounding radius (body = (1-2r)-cube (+) ball(r))
// Penalty = sum_pairs sum_v relu(Bh - w.Vi + m')^2 + relu(w.Vj - Bh + m')^2
//         + sum_i,v,d relu(V - s + m')^2 + relu(-V + m')^2,     m' = margin + r.
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define NV 16

static void quatL(const double *q, double M[4][4]) {
    double w = q[0], x = q[1], y = q[2], z = q[3];
    double T[4][4] = {{w, -x, -y, -z}, {x, w, -z, y}, {y, z, w, -x}, {z, -y, x, w}};
    memcpy(M, T, sizeof(T));
}
static void quatR(const double *q, double M[4][4]) {
    double w = q[0], x = q[1], y = q[2], z = q[3];
    double T[4][4] = {{w, -x, -y, -z}, {x, w, z, -y}, {y, -z, w, x}, {z, y, -x, w}};
    memcpy(M, T, sizeof(T));
}

// evaluate penalty (and gradient if g* != NULL) for one configuration
static double eval_one(int n, const double *c, const double *p, const double *W, const double *Bh,
                       double s, double r, double margin, int prune,
                       double *gc, double *gp, double *gW, double *gB, double *mx_out,
                       double *Rbuf, double *Vbuf, double *gV, double *abuf) {
    const double mp = margin + r, h = 0.5 * (1.0 - 2.0 * r);
    double pen = 0.0, mx = 0.0;
    // rotations and vertices
    for (int i = 0; i < n; i++) {
        const double *pa = p + 8 * i, *pb = pa + 4;
        double na = sqrt(pa[0]*pa[0] + pa[1]*pa[1] + pa[2]*pa[2] + pa[3]*pa[3]);
        double nb = sqrt(pb[0]*pb[0] + pb[1]*pb[1] + pb[2]*pb[2] + pb[3]*pb[3]);
        double *A = abuf + 10 * i;                 // a (4), b (4), |a|, |b|
        for (int t = 0; t < 4; t++) { A[t] = pa[t] / na; A[4 + t] = pb[t] / nb; }
        A[8] = na; A[9] = nb;
        double La[4][4], Rb[4][4];
        quatL(A, La); quatR(A + 4, Rb);
        double *R = Rbuf + 16 * i;
        for (int k = 0; k < 4; k++) for (int l = 0; l < 4; l++) {
            double acc = 0; for (int m = 0; m < 4; m++) acc += La[k][m] * Rb[m][l]; R[4 * k + l] = acc;
        }
        for (int v = 0; v < NV; v++) {
            double sg[4] = {(v & 1) ? h : -h, (v & 2) ? h : -h, (v & 4) ? h : -h, (v & 8) ? h : -h};
            for (int d = 0; d < 4; d++)
                Vbuf[(i * NV + v) * 4 + d] = c[4 * i + d] + R[4 * d + 0] * sg[0] + R[4 * d + 1] * sg[1] + R[4 * d + 2] * sg[2] + R[4 * d + 3] * sg[3];
        }
    }
    if (gV) memset(gV, 0, sizeof(double) * n * NV * 4);
    // pairs
    int k = 0;
    const double far2 = (2.0 + 2.0 * mp + 0.05) * (2.0 + 2.0 * mp + 0.05);
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++, k++) {
        const double *Wk = W + 4 * k;
        if (prune) {
            double dd = 0; for (int d = 0; d < 4; d++) { double e = c[4*i+d] - c[4*j+d]; dd += e * e; }
            if (dd > far2) { if (gW) { memset(gW + 4 * k, 0, 4 * sizeof(double)); gB[k] = 0; } continue; }
        }
        double nw = sqrt(Wk[0]*Wk[0] + Wk[1]*Wk[1] + Wk[2]*Wk[2] + Wk[3]*Wk[3]);
        double w[4] = {Wk[0] / nw, Wk[1] / nw, Wk[2] / nw, Wk[3] / nw};
        double gw[4] = {0, 0, 0, 0}, gb = 0, bk = Bh[k];
        for (int v = 0; v < NV; v++) {
            const double *Vi = Vbuf + (i * NV + v) * 4, *Vj = Vbuf + (j * NV + v) * 4;
            double al = w[0]*Vi[0] + w[1]*Vi[1] + w[2]*Vi[2] + w[3]*Vi[3];
            double be = w[0]*Vj[0] + w[1]*Vj[1] + w[2]*Vj[2] + w[3]*Vj[3];
            double ta = bk - al + mp, tb = be - bk + mp;
            if (ta > 0) {
                pen += ta * ta; if (ta > mx) mx = ta;
                if (gV) { double g = -2 * ta; gb -= g;
                    for (int d = 0; d < 4; d++) { gV[(i * NV + v) * 4 + d] += g * w[d]; gw[d] += g * Vi[d]; } }
            }
            if (tb > 0) {
                pen += tb * tb; if (tb > mx) mx = tb;
                if (gV) { double g = 2 * tb; gb -= g;
                    for (int d = 0; d < 4; d++) { gV[(j * NV + v) * 4 + d] += g * w[d]; gw[d] += g * Vj[d]; } }
            }
        }
        if (gW) {   // through normalisation w = W/|W|
            double dt = gw[0]*w[0] + gw[1]*w[1] + gw[2]*w[2] + gw[3]*w[3];
            for (int d = 0; d < 4; d++) gW[4 * k + d] = (gw[d] - dt * w[d]) / nw;
            gB[k] = gb;
        }
    }
    // container
    for (int i = 0; i < n; i++) for (int v = 0; v < NV; v++) for (int d = 0; d < 4; d++) {
        double x = Vbuf[(i * NV + v) * 4 + d];
        double t1 = x - s + mp, t2 = -x + mp;
        if (t1 > 0) { pen += t1 * t1; if (t1 > mx) mx = t1; if (gV) gV[(i * NV + v) * 4 + d] += 2 * t1; }
        if (t2 > 0) { pen += t2 * t2; if (t2 > mx) mx = t2; if (gV) gV[(i * NV + v) * 4 + d] -= 2 * t2; }
    }
    if (mx_out) *mx_out = mx;
    if (!gV) return pen;
    // fold vertex gradients into centres and rotation parameters
    for (int i = 0; i < n; i++) {
        double G[4][4] = {{0}};
        for (int d = 0; d < 4; d++) gc[4 * i + d] = 0;
        for (int v = 0; v < NV; v++) {
            double sg[4] = {(v & 1) ? h : -h, (v & 2) ? h : -h, (v & 4) ? h : -h, (v & 8) ? h : -h};
            const double *g = gV + (i * NV + v) * 4;
            for (int d = 0; d < 4; d++) { gc[4 * i + d] += g[d]; for (int l = 0; l < 4; l++) G[d][l] += g[d] * sg[l]; }
        }
        const double *A = abuf + 10 * i;
        double La[4][4], Rb[4][4], M[4][4], N[4][4];
        quatL(A, La); quatR(A + 4, Rb);
        for (int x = 0; x < 4; x++) for (int y = 0; y < 4; y++) {     // M = G Rb^T, N = La^T G
            double m1 = 0, m2 = 0;
            for (int z = 0; z < 4; z++) { m1 += G[x][z] * Rb[y][z]; m2 += La[z][x] * G[z][y]; }
            M[x][y] = m1; N[x][y] = m2;
        }
        double ga[4] = {M[0][0] + M[1][1] + M[2][2] + M[3][3],
                        -M[0][1] + M[1][0] - M[2][3] + M[3][2],
                        -M[0][2] + M[1][3] + M[2][0] - M[3][1],
                        -M[0][3] - M[1][2] + M[2][1] + M[3][0]};
        double gq[4] = {N[0][0] + N[1][1] + N[2][2] + N[3][3],
                        -N[0][1] + N[1][0] + N[2][3] - N[3][2],
                        -N[0][2] - N[1][3] + N[2][0] + N[3][1],
                        -N[0][3] + N[1][2] - N[2][1] + N[3][0]};
        // through normalisation a = pa/|pa|
        double da = ga[0]*A[0] + ga[1]*A[1] + ga[2]*A[2] + ga[3]*A[3];
        double db = gq[0]*A[4] + gq[1]*A[5] + gq[2]*A[6] + gq[3]*A[7];
        for (int t = 0; t < 4; t++) {
            gp[8 * i + t] = (ga[t] - da * A[t]) / A[8];
            gp[8 * i + 4 + t] = (gq[t] - db * A[4 + t]) / A[9];
        }
    }
    return pen;
}

static inline void adam(double *x, const double *g, double *m, double *v, int len, double lr, double bc1, double bc2) {
    const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
    for (int t = 0; t < len; t++) {
        m[t] = b1 * m[t] + (1 - b1) * g[t];
        v[t] = b2 * v[t] + (1 - b2) * g[t] * g[t];
        x[t] -= lr * (m[t] / bc1) / (sqrt(v[t] / bc2) + eps);
    }
}

// Run `steps` Adam steps on all B configurations; then write penalty and max violation.
// state = [mc, vc, mp, vp, mW, vW, mB, vB] laid out like the parameters. *t is Adam's step counter.
int run_steps(int B, int n, double *c, double *p, double *W, double *Bh, const double *s, const double *r,
              double margin, double **state, long *t, int steps, double lr, int prune,
              double *pen_out, double *mx_out) {
    int P = n * (n - 1) / 2;
    double *gc = malloc(sizeof(double) * 4 * n), *gp = malloc(sizeof(double) * 8 * n);
    double *gW = malloc(sizeof(double) * 4 * (P ? P : 1)), *gB = malloc(sizeof(double) * (P ? P : 1));
    double *Rbuf = malloc(sizeof(double) * 16 * n), *Vbuf = malloc(sizeof(double) * n * NV * 4);
    double *gV = malloc(sizeof(double) * n * NV * 4), *abuf = malloc(sizeof(double) * 10 * n);
    for (int st = 0; st < steps; st++) {
        (*t)++;
        double bc1 = 1 - pow(0.9, (double)*t), bc2 = 1 - pow(0.999, (double)*t);
        for (int b = 0; b < B; b++) {
            double *cb = c + (long)b * 4 * n, *pb = p + (long)b * 8 * n, *Wb = W + (long)b * 4 * P, *Bb = Bh + (long)b * P;
            eval_one(n, cb, pb, Wb, Bb, s[b], r[b], margin, prune, gc, gp, gW, gB, NULL, Rbuf, Vbuf, gV, abuf);
            adam(cb, gc, state[0] + (long)b * 4 * n, state[1] + (long)b * 4 * n, 4 * n, lr, bc1, bc2);
            adam(pb, gp, state[2] + (long)b * 8 * n, state[3] + (long)b * 8 * n, 8 * n, lr, bc1, bc2);
            adam(Wb, gW, state[4] + (long)b * 4 * P, state[5] + (long)b * 4 * P, 4 * P, lr, bc1, bc2);
            adam(Bb, gB, state[6] + (long)b * P, state[7] + (long)b * P, P, lr, bc1, bc2);
        }
    }
    for (int b = 0; b < B; b++) {
        double mx;
        pen_out[b] = eval_one(n, c + (long)b * 4 * n, p + (long)b * 8 * n, W + (long)b * 4 * P, Bh + (long)b * P, s[b], r[b],
                              margin, prune, NULL, NULL, NULL, NULL, &mx, Rbuf, Vbuf, NULL, abuf);
        mx_out[b] = mx;
    }
    free(gc); free(gp); free(gW); free(gB); free(Rbuf); free(Vbuf); free(gV); free(abuf);
    return 0;
}

// gradient only (for verification against torch autograd)
double grad_one(int n, const double *c, const double *p, const double *W, const double *Bh, double s, double r,
                double margin, int prune, double *gc, double *gp, double *gW, double *gB) {
    int P = n * (n - 1) / 2;
    double *Rbuf = malloc(sizeof(double) * 16 * n), *Vbuf = malloc(sizeof(double) * n * NV * 4);
    double *gV = malloc(sizeof(double) * n * NV * 4), *abuf = malloc(sizeof(double) * 10 * n);
    (void)P;
    double pen = eval_one(n, c, p, W, Bh, s, r, margin, prune, gc, gp, gW, gB, NULL, Rbuf, Vbuf, gV, abuf);
    free(Rbuf); free(Vbuf); free(gV); free(abuf);
    return pen;
}

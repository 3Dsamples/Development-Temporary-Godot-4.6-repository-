// File: insert_after_line_X in exact_arithmetic.h – replaces all placeholder functions.

// ----------------------------------------------------------------------
// Generic helper to add the expansion of a*b to an existing expansion arr/len
// ----------------------------------------------------------------------
inline void add_product_to_expansion(double* arr, int& len, double a, double b) noexcept {
    double prod[2]; int plen = 2;
    two_product(a, b, prod[0], prod[1]);
    double sum[16]; int sumlen;
    fast_expansion_sum(arr, len, prod, plen, sum, sumlen);
    for (int i = 0; i < sumlen; ++i) arr[i] = sum[i];
    len = sumlen;
}

// ----------------------------------------------------------------------
// Exact 2D cross product (ux*vy - uy*vx) with all tail components
// ----------------------------------------------------------------------
inline void exact_cross2(const double* ux, const double* uy,
                         const double* vx, const double* vy,
                         double* cross, int& crosslen) noexcept {
    double a[8], b[8]; int alen = 0, blen = 0;
    // a = ux * vy
    two_product(ux[0], vy[0], a[0], a[1]); alen = 2;
    if (ux[1] != 0.0 || vy[1] != 0.0) {
        if (ux[1] != 0.0) {
            add_product_to_expansion(a, alen, ux[1], vy[0]);
            if (vy[1] != 0.0) add_product_to_expansion(a, alen, ux[1], vy[1]);
        }
        if (vy[1] != 0.0) add_product_to_expansion(a, alen, ux[0], vy[1]);
    }
    // b = uy * vx
    two_product(uy[0], vx[0], b[0], b[1]); blen = 2;
    if (uy[1] != 0.0 || vx[1] != 0.0) {
        if (uy[1] != 0.0) {
            add_product_to_expansion(b, blen, uy[1], vx[0]);
            if (vx[1] != 0.0) add_product_to_expansion(b, blen, uy[1], vx[1]);
        }
        if (vx[1] != 0.0) add_product_to_expansion(b, blen, uy[0], vx[1]);
    }
    // cross = a - b
    double neg_b[8]; for (int i = 0; i < blen; ++i) neg_b[i] = -b[i];
    fast_expansion_sum(a, alen, neg_b, blen, cross, crosslen);
}

// ----------------------------------------------------------------------
// Exact squared distance (dx^2 + dy^2) with all tail components
// ----------------------------------------------------------------------
inline void exact_sqdist2(const double* dx, const double* dy,
                          double* dist, int& distlen) noexcept {
    double dx2[8], dy2[8]; int dx2len = 0, dy2len = 0;

    // dx2 = dx*dx
    two_product(dx[0], dx[0], dx2[0], dx2[1]); dx2len = 2;
    if (dx[1] != 0.0) {
        add_product_to_expansion(dx2, dx2len, dx[1], dx[1]);           // dx1^2
        add_product_to_expansion(dx2, dx2len, 2.0 * dx[0], dx[1]);    // 2*dx0*dx1
    }

    // dy2 = dy*dy
    two_product(dy[0], dy[0], dy2[0], dy2[1]); dy2len = 2;
    if (dy[1] != 0.0) {
        add_product_to_expansion(dy2, dy2len, dy[1], dy[1]);           // dy1^2
        add_product_to_expansion(dy2, dy2len, 2.0 * dy[0], dy[1]);    // 2*dy0*dy1
    }

    fast_expansion_sum(dx2, dx2len, dy2, dy2len, dist, distlen);
}

// ----------------------------------------------------------------------
// Exact 3D squared distance (dx^2 + dy^2 + dz^2)
// ----------------------------------------------------------------------
inline void exact_sqdist3(const double* dx, const double* dy, const double* dz,
                          double* dist, int& distlen) noexcept {
    double dx2[8], dy2[8], dz2[8]; int dx2len = 0, dy2len = 0, dz2len = 0;

    // dx2
    two_product(dx[0], dx[0], dx2[0], dx2[1]); dx2len = 2;
    if (dx[1] != 0.0) {
        add_product_to_expansion(dx2, dx2len, dx[1], dx[1]);
        add_product_to_expansion(dx2, dx2len, 2.0 * dx[0], dx[1]);
    }
    // dy2
    two_product(dy[0], dy[0], dy2[0], dy2[1]); dy2len = 2;
    if (dy[1] != 0.0) {
        add_product_to_expansion(dy2, dy2len, dy[1], dy[1]);
        add_product_to_expansion(dy2, dy2len, 2.0 * dy[0], dy[1]);
    }
    // dz2
    two_product(dz[0], dz[0], dz2[0], dz2[1]); dz2len = 2;
    if (dz[1] != 0.0) {
        add_product_to_expansion(dz2, dz2len, dz[1], dz[1]);
        add_product_to_expansion(dz2, dz2len, 2.0 * dz[0], dz[1]);
    }

    double tmp[16]; int tmplen;
    fast_expansion_sum(dx2, dx2len, dy2, dy2len, tmp, tmplen);
    fast_expansion_sum(tmp, tmplen, dz2, dz2len, dist, distlen);
}

// ----------------------------------------------------------------------
// Exact 3D cross product (a · (b × c)) for three vectors given by expansions
// ----------------------------------------------------------------------
inline void exact_cross3(const double* ux, const double* uy, const double* uz,
                         const double* vx, const double* vy, const double* vz,
                         const double* wx, const double* wy, const double* wz,
                         double* result, int& reslen) noexcept {
    // Compute u · (v × w) = ux*(vy*wz - vz*wy) + uy*(vz*wx - vx*wz) + uz*(vx*wy - vy*wx)
    // We'll compute the three components separately and sum them.

    // Helper to compute exact (a*b - c*d) and add tails
    auto exact_cross2_pair = [&](const double* a, const double* b,
                                 const double* c, const double* d,
                                 double* out, int& outlen) {
        double ab[8], cd[8]; int ablen = 0, cdlen = 0;
        // a*b
        two_product(a[0], b[0], ab[0], ab[1]); ablen = 2;
        if (a[1] != 0.0 || b[1] != 0.0) {
            if (a[1] != 0.0) {
                add_product_to_expansion(ab, ablen, a[1], b[0]);
                if (b[1] != 0.0) add_product_to_expansion(ab, ablen, a[1], b[1]);
            }
            if (b[1] != 0.0) add_product_to_expansion(ab, ablen, a[0], b[1]);
        }
        // c*d
        two_product(c[0], d[0], cd[0], cd[1]); cdlen = 2;
        if (c[1] != 0.0 || d[1] != 0.0) {
            if (c[1] != 0.0) {
                add_product_to_expansion(cd, cdlen, c[1], d[0]);
                if (d[1] != 0.0) add_product_to_expansion(cd, cdlen, c[1], d[1]);
            }
            if (d[1] != 0.0) add_product_to_expansion(cd, cdlen, c[0], d[1]);
        }
        // ab - cd
        double neg_cd[8]; for (int i = 0; i < cdlen; ++i) neg_cd[i] = -cd[i];
        fast_expansion_sum(ab, ablen, neg_cd, cdlen, out, outlen);
    };

    // Component 1: ux * (vy*wz - vz*wy)
    double cross_yz[16]; int cyzlen;
    exact_cross2_pair(vy, wz, vz, wy, cross_yz, cyzlen);
    double comp1[16]; int comp1len = 0;
    scale_expansion(cross_yz, cyzlen, ux[0], comp1, comp1len);
    if (ux[1] != 0.0) {
        double tail[16]; int taillen;
        scale_expansion(cross_yz, cyzlen, ux[1], tail, taillen);
        double sum[32]; int sumlen;
        fast_expansion_sum(comp1, comp1len, tail, taillen, sum, sumlen);
        for (int i = 0; i < sumlen; ++i) comp1[i] = sum[i];
        comp1len = sumlen;
    }

    // Component 2: uy * (vz*wx - vx*wz)
    double cross_zx[16]; int czxlen;
    exact_cross2_pair(vz, wx, vx, wz, cross_zx, czxlen);
    double comp2[16]; int comp2len = 0;
    scale_expansion(cross_zx, czxlen, uy[0], comp2, comp2len);
    if (uy[1] != 0.0) {
        double tail[16]; int taillen;
        scale_expansion(cross_zx, czxlen, uy[1], tail, taillen);
        double sum[32]; int sumlen;
        fast_expansion_sum(comp2, comp2len, tail, taillen, sum, sumlen);
        for (int i = 0; i < sumlen; ++i) comp2[i] = sum[i];
        comp2len = sumlen;
    }

    // Component 3: uz * (vx*wy - vy*wx)
    double cross_xy[16]; int cxylen;
    exact_cross2_pair(vx, wy, vy, wx, cross_xy, cxylen);
    double comp3[16]; int comp3len = 0;
    scale_expansion(cross_xy, cxylen, uz[0], comp3, comp3len);
    if (uz[1] != 0.0) {
        double tail[16]; int taillen;
        scale_expansion(cross_xy, cxylen, uz[1], tail, taillen);
        double sum[32]; int sumlen;
        fast_expansion_sum(comp3, comp3len, tail, taillen, sum, sumlen);
        for (int i = 0; i < sumlen; ++i) comp3[i] = sum[i];
        comp3len = sumlen;
    }

    // Sum all three components
    double tmp[32]; int tmplen;
    fast_expansion_sum(comp1, comp1len, comp2, comp2len, tmp, tmplen);
    fast_expansion_sum(tmp, tmplen, comp3, comp3len, result, reslen);
}

// ----------------------------------------------------------------------
// Complete incircleadapt using exact squared distances and 2D cross products
// ----------------------------------------------------------------------
inline double incircleadapt(const double* pa, const double* pb,
                            const double* pc, const double* pd,
                            double permanent) noexcept {
    double adx[2], ady[2], bdx[2], bdy[2], cdx[2], cdy[2];
    two_diff(pa[0], pd[0], adx[0], adx[1]);
    two_diff(pa[1], pd[1], ady[0], ady[1]);
    two_diff(pb[0], pd[0], bdx[0], bdx[1]);
    two_diff(pb[1], pd[1], bdy[0], bdy[1]);
    two_diff(pc[0], pd[0], cdx[0], cdx[1]);
    two_diff(pc[1], pd[1], cdy[0], cdy[1]);

    double adist[8], bdist[8], cdist[8];
    int adlen, bdlen, cdlen;
    exact_sqdist2(adx, ady, adist, adlen);
    exact_sqdist2(bdx, bdy, bdist, bdlen);
    exact_sqdist2(cdx, cdy, cdist, cdlen);

    double ab[8], bc[8], ca[8];
    int ablen, bclen, calen;
    exact_cross2(adx, ady, bdx, bdy, ab, ablen);
    exact_cross2(bdx, bdy, cdx, cdy, bc, bclen);
    exact_cross2(cdx, cdy, adx, ady, ca, calen);

    // incircle = alen2*bc - blen2*ca + clen2*ab
    double term1[16], term2[16], term3[16];
    int len1, len2, len3;
    scale_expansion(bc, bclen, adist, adlen, term1, len1);
    scale_expansion(ca, calen, bdist, bdlen, term2, len2);
    scale_expansion(ab, ablen, cdist, cdlen, term3, len3);

    // subtract term2
    double neg_term2[16]; for (int i = 0; i < len2; ++i) neg_term2[i] = -term2[i];
    double tmp[24]; int tmplen;
    fast_expansion_sum(term1, len1, neg_term2, len2, tmp, tmplen);
    double det[32]; int detlen;
    fast_expansion_sum(tmp, tmplen, term3, len3, det, detlen);
    compress(det, detlen, det, detlen);

    double err = 0.0;
    for (int i = 0; i < detlen; ++i) err += std::fabs(det[i]);
    if (std::fabs(det[detlen - 1]) >= 1.776e-14 * permanent)
        return det[detlen - 1];
    return 0.0; // zero within error bound
}

// ----------------------------------------------------------------------
// Complete insphereadapt using exact squared distances and 3D cross products
// ----------------------------------------------------------------------
inline double insphereadapt(const double* pa, const double* pb,
                            const double* pc, const double* pd,
                            const double* pe, double permanent) noexcept {
    double adx[2], ady[2], adz[2];
    double bdx[2], bdy[2], bdz[2];
    double cdx[2], cdy[2], cdz[2];
    double ddx[2], ddy[2], ddz[2];
    two_diff(pa[0], pe[0], adx[0], adx[1]);
    two_diff(pa[1], pe[1], ady[0], ady[1]);
    two_diff(pa[2], pe[2], adz[0], adz[1]);
    two_diff(pb[0], pe[0], bdx[0], bdx[1]);
    two_diff(pb[1], pe[1], bdy[0], bdy[1]);
    two_diff(pb[2], pe[2], bdz[0], bdz[1]);
    two_diff(pc[0], pe[0], cdx[0], cdx[1]);
    two_diff(pc[1], pe[1], cdy[0], cdy[1]);
    two_diff(pc[2], pe[2], cdz[0], cdz[1]);
    two_diff(pd[0], pe[0], ddx[0], ddx[1]);
    two_diff(pd[1], pe[1], ddy[0], ddy[1]);
    two_diff(pd[2], pe[2], ddz[0], ddz[1]);

    double adist[16], bdist[16], cdist[16], ddist[16];
    int adlen, bdlen, cdlen, ddlen;
    exact_sqdist3(adx, ady, adz, adist, adlen);
    exact_sqdist3(bdx, bdy, bdz, bdist, bdlen);
    exact_sqdist3(cdx, cdy, cdz, cdist, cdlen);
    exact_sqdist3(ddx, ddy, ddz, ddist, ddlen);

    double bcd[16], cda[16], dab[16], abc[16];
    int bcdlen, cdalen, dablen, abclen;
    exact_cross3(bdx, bdy, bdz, cdx, cdy, cdz, ddx, ddy, ddz, bcd, bcdlen);
    exact_cross3(cdx, cdy, cdz, ddx, ddy, ddz, adx, ady, adz, cda, cdalen);
    exact_cross3(ddx, ddy, ddz, adx, ady, adz, bdx, bdy, bdz, dab, dablen);
    exact_cross3(adx, ady, adz, bdx, bdy, bdz, cdx, cdy, cdz, abc, abclen);

    // insphere = adist * bcd - bdist * cda + cdist * dab - ddist * abc
    double term1[32], term2[32], term3[32], term4[32];
    int len1, len2, len3, len4;
    scale_expansion(bcd, bcdlen, adist, adlen, term1, len1);
    scale_expansion(cda, cdalen, bdist, bdlen, term2, len2);
    scale_expansion(dab, dablen, cdist, cdlen, term3, len3);
    scale_expansion(abc, abclen, ddist, ddlen, term4, len4);

    // combine: term1 - term2 + term3 - term4
    double neg_term2[32]; for (int i = 0; i < len2; ++i) neg_term2[i] = -term2[i];
    double tmp1[48]; int tmplen1;
    fast_expansion_sum(term1, len1, neg_term2, len2, tmp1, tmplen1);
    double tmp2[48]; int tmplen2;
    fast_expansion_sum(tmp1, tmplen1, term3, len3, tmp2, tmplen2);
    double neg_term4[32]; for (int i = 0; i < len4; ++i) neg_term4[i] = -term4[i];
    double det[64]; int detlen;
    fast_expansion_sum(tmp2, tmplen2, neg_term4, len4, det, detlen);
    compress(det, detlen, det, detlen);

    double err = 0.0;
    for (int i = 0; i < detlen; ++i) err += std::fabs(det[i]);
    if (std::fabs(det[detlen - 1]) >= 8.88e-14 * permanent)
        return det[detlen - 1];
    return 0.0;
}
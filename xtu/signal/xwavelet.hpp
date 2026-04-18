// signal/xwavelet.hpp

#ifndef XTENSOR_XWAVELET_HPP
#define XTENSOR_XWAVELET_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../math/xlinalg.hpp"
#include "../math/xstats.hpp"
#include "../math/xnorm.hpp"
#include "lfilter.hpp"
#include "xwindows.hpp"

#include <cmath>
#include <vector>
#include <array>
#include <string>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <functional>
#include <type_traits>
#include <complex>
#include <map>
#include <tuple>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace wavelet
        {
            // --------------------------------------------------------------------
            // Wavelet filter bank structure
            // --------------------------------------------------------------------
            struct Wavelet
            {
                std::string name;
                std::vector<double> dec_lo;   // decomposition low-pass filter
                std::vector<double> dec_hi;   // decomposition high-pass filter
                std::vector<double> rec_lo;   // reconstruction low-pass filter
                std::vector<double> rec_hi;   // reconstruction high-pass filter
                int vanishing_moments_psi = 0;
                int vanishing_moments_phi = 0;
                bool orthogonal = true;
                bool biorthogonal = false;
                bool symmetric = false;
            };

            namespace detail
            {
                // --------------------------------------------------------------------
                // Helper: Convolve and downsample (decimation by 2)
                // --------------------------------------------------------------------
                template <class T>
                std::vector<T> convolve_down(const std::vector<T>& x, const std::vector<T>& filter)
                {
                    size_t n = x.size();
                    size_t f_len = filter.size();
                    size_t out_len = (n + f_len - 1) / 2;
                    std::vector<T> result(out_len, 0.0);
                    for (size_t i = 0; i < out_len; ++i)
                    {
                        T sum = 0;
                        for (size_t j = 0; j < f_len; ++j)
                        {
                            int idx = 2 * static_cast<int>(i) - static_cast<int>(j) + static_cast<int>(f_len) - 1;
                            if (idx >= 0 && idx < static_cast<int>(n))
                                sum += x[static_cast<size_t>(idx)] * filter[j];
                        }
                        result[i] = sum;
                    }
                    return result;
                }

                // --------------------------------------------------------------------
                // Helper: Upsample and convolve (interpolation)
                // --------------------------------------------------------------------
                template <class T>
                std::vector<T> upsample_convolve(const std::vector<T>& x, const std::vector<T>& filter, size_t target_len)
                {
                    size_t n = x.size();
                    size_t f_len = filter.size();
                    std::vector<T> result(target_len, 0.0);
                    for (size_t i = 0; i < target_len; ++i)
                    {
                        T sum = 0;
                        for (size_t j = 0; j < f_len; ++j)
                        {
                            int idx = static_cast<int>(i) - static_cast<int>(j);
                            if (idx >= 0 && idx % 2 == 0)
                            {
                                size_t x_idx = static_cast<size_t>(idx / 2);
                                if (x_idx < n)
                                    sum += x[x_idx] * filter[j];
                            }
                        }
                        result[i] = sum;
                    }
                    return result;
                }

                // --------------------------------------------------------------------
                // Periodized extension (for circular convolution)
                // --------------------------------------------------------------------
                template <class T>
                T periodized_get(const std::vector<T>& x, int idx)
                {
                    int n = static_cast<int>(x.size());
                    idx = ((idx % n) + n) % n;
                    return x[static_cast<size_t>(idx)];
                }

                template <class T>
                std::vector<T> per_convolve_down(const std::vector<T>& x, const std::vector<T>& filter)
                {
                    size_t n = x.size();
                    size_t f_len = filter.size();
                    size_t out_len = (n + 1) / 2;  // for odd length, keep ceil
                    std::vector<T> result(out_len, 0.0);
                    for (size_t i = 0; i < out_len; ++i)
                    {
                        T sum = 0;
                        for (size_t j = 0; j < f_len; ++j)
                        {
                            int idx = 2 * static_cast<int>(i) - static_cast<int>(j);
                            sum += periodized_get(x, idx) * filter[j];
                        }
                        result[i] = sum;
                    }
                    return result;
                }

                template <class T>
                std::vector<T> per_upsample_convolve(const std::vector<T>& x, const std::vector<T>& filter, size_t target_len)
                {
                    size_t n = x.size();
                    size_t f_len = filter.size();
                    std::vector<T> result(target_len, 0.0);
                    for (size_t i = 0; i < target_len; ++i)
                    {
                        T sum = 0;
                        for (size_t j = 0; j < f_len; ++j)
                        {
                            int idx = static_cast<int>(i) - static_cast<int>(j);
                            if (idx % 2 == 0)
                            {
                                sum += periodized_get(x, idx / 2) * filter[j];
                            }
                        }
                        result[i] = sum;
                    }
                    return result;
                }

                // --------------------------------------------------------------------
                // Orthogonal filter construction from scaling filter
                // --------------------------------------------------------------------
                inline std::vector<double> qmf(const std::vector<double>& lo)
                {
                    // Quadrature mirror filter: hi[k] = (-1)^k * lo[L-1-k]
                    size_t L = lo.size();
                    std::vector<double> hi(L);
                    for (size_t i = 0; i < L; ++i)
                    {
                        hi[i] = (i % 2 == 0 ? 1.0 : -1.0) * lo[L - 1 - i];
                    }
                    return hi;
                }

                inline std::vector<double> reverse_filter(const std::vector<double>& f)
                {
                    return std::vector<double>(f.rbegin(), f.rend());
                }

                // --------------------------------------------------------------------
                // Wavelet coefficient generation functions
                // --------------------------------------------------------------------
                
                // Daubechies wavelets (dbN) for N=1..20
                std::vector<double> daubechies_scaling_filter(int N)
                {
                    // Coefficients for db1 (Haar), db2, db3, db4...
                    // Precomputed for common orders
                    static const std::map<int, std::vector<double>> db_coeffs = {
                        {1, {0.7071067811865475, 0.7071067811865475}},  // Haar
                        {2, {0.4829629131445341, 0.8365163037378079, 0.2241438680420134, -0.1294095225512604}},
                        {3, {0.3326705529500826, 0.8068915093110925, 0.4598775021184915, -0.1350110200102546, -0.0854412738820267, 0.0352262918857095}},
                        {4, {0.2303778133088965, 0.7148465705529154, 0.6308807679398587, -0.0279837694168599, -0.1870348117190931, 0.0308413818355607, 0.0328830116668852, -0.0105974017850690}},
                        {5, {0.1601023979741929, 0.6038292697971895, 0.7243085284377726, 0.1384281459013203, -0.2422948870663824, -0.0322448695846381, 0.0775714938400459, -0.0062414902127983, -0.0125807519990820, 0.0033357252854738}},
                        {6, {0.1115407433501095, 0.4946238903984531, 0.7511339080210954, 0.3152503517091982, -0.2262646939654400, -0.1297668675672625, 0.0975016055873225, 0.0275228655303053, -0.0315820393174862, 0.0005538422011614, 0.0047772575109455, -0.0010773010853085}},
                        {7, {0.0778520540850037, 0.3965393194818912, 0.7291320908461957, 0.4697822874051889, -0.1439060039285649, -0.2240361849938749, 0.0713092192668272, 0.0806126091510774, -0.0380299369350104, -0.0165745416306655, 0.0125509985560986, 0.0004295779729214, -0.0018016407040474, 0.0003537137999745}},
                        {8, {0.0544158422431072, 0.3128715909143166, 0.6756307362973195, 0.5853546836542159, -0.0158291052563823, -0.2840155429615824, 0.0004724845739124, 0.1287474266204893, -0.0173693010018090, -0.0440882539307971, 0.0139810279174001, 0.0087460940474065, -0.0048703529934518, -0.0003917403733770, 0.0006754494064506, -0.0001174767841248}}
                    };
                    auto it = db_coeffs.find(N);
                    if (it != db_coeffs.end())
                        return it->second;
                    XTENSOR_THROW(std::invalid_argument, "Daubechies wavelet of order " + std::to_string(N) + " not precomputed. Use N<=8.");
                    return {};
                }

                // Symlets (symN) - symmetric version of Daubechies
                std::vector<double> symlets_scaling_filter(int N)
                {
                    // Precomputed for sym2..sym8
                    static const std::map<int, std::vector<double>> sym_coeffs = {
                        {2, {0.4829629131445341, 0.8365163037378079, 0.2241438680420134, -0.1294095225512604}},
                        {3, {0.3326705529500826, 0.8068915093110925, 0.4598775021184915, -0.1350110200102546, -0.0854412738820267, 0.0352262918857095}},
                        {4, {-0.0757657147893411, -0.0296355276459541, 0.4976186676324578, 0.8037387518052160, 0.2978577956055426, -0.0992195435769353, -0.0126039672622612, 0.0322231006040713}},
                        {5, {0.0273330683451645, 0.0295194909260734, -0.0391342493026494, 0.1993975339769956, 0.7234076904038076, 0.6339789634569490, 0.0166021057643860, -0.1753280899081075, -0.0211018340249302, 0.0195388827353869}},
                        {6, {0.0154041093273377, 0.0034907120843304, -0.1179901111484105, -0.0483117425859981, 0.4910559419276396, 0.7876411410287941, 0.3379294217282401, -0.0726375227866000, -0.0210602925126955, 0.0447249017707482, 0.0017677118643981, -0.0078007083247650}},
                        {7, {0.0026818145681874, -0.0010473848889657, -0.0126363034031636, 0.0305155131659062, 0.0678926935015971, -0.0495528349370410, 0.0174412550878319, 0.5361019170907720, 0.7677643170048758, 0.2886296317522300, -0.1400472404427038, -0.1078082377036168, 0.0040102448717034, 0.0102681767082552}},
                        {8, {0.0018899503329007, -0.0003029205145516, -0.0149522583367926, 0.0038087520140602, 0.0491371796734763, -0.0272190299168137, -0.0519458381078751, 0.3644418948351784, 0.7771857516997478, 0.4813596512592012, -0.0612733590679088, -0.1432942383510542, 0.0076074873252847, 0.0316950878103452, -0.0005421323313697, -0.0033824159510024}}
                    };
                    auto it = sym_coeffs.find(N);
                    if (it != sym_coeffs.end())
                        return it->second;
                    XTENSOR_THROW(std::invalid_argument, "Symlet of order " + std::to_string(N) + " not precomputed.");
                    return {};
                }

                // Coiflets (coifN)
                std::vector<double> coiflet_scaling_filter(int N)
                {
                    static const std::map<int, std::vector<double>> coif_coeffs = {
                        {1, {-0.0156557281354647, -0.0727326195128539, 0.3848648468642029, 0.8525720202122554, 0.3378976624578092, -0.0727326195128539}},
                        {2, {-0.0007205494453693, -0.0018232088707110, 0.0056114348193652, 0.0236801719461965, -0.0594344186464712, -0.0764885990784873, 0.4170051844237773, 0.8127236354493967, 0.3861100668233679, -0.0673725547220195, -0.0414649367819662, 0.0163873364635998}},
                        {3, {0.0000370998238834, -0.0002535611322392, -0.0010182098221261, 0.0043409378258106, 0.0051062632491751, -0.0264397904354517, -0.0171222497699125, 0.0973967122557091, 0.0358381154160370, -0.1877355236843138, -0.0582626920638929, 0.2693177826031112, 0.6885833778312518, 0.4758527053314997, 0.0721073682587302, -0.0720776311500117, -0.0302032909760493, 0.0148976350763354}},
                        {4, {0.0000333660435925, 0.0000822094331135, -0.0002560220402357, -0.0009946060553239, 0.0014579538569198, 0.0064847154605097, -0.0038848369167977, -0.0265003384816036, 0.0029947212523340, 0.0733162133501086, 0.0081139737065683, -0.1493703042861253, -0.0449113634637051, 0.2384292331355981, 0.6883254723389560, 0.4976846703111483, 0.0370438936796877, -0.0990309154258712, -0.0213623231457004, 0.0271077320112307, 0.0058100135231264, -0.0043380150608126, -0.0006954233422415, 0.0004041129887716}},
                        {5, {0.0000045215643835, -0.0000107717122187, -0.0000645213562712, 0.0001379789441148, 0.0005062613716933, -0.0011124674500539, -0.0025794967353968, 0.0060597805072154, 0.0094014724450152, -0.0241269205598326, -0.0246623808554984, 0.0728663273750566, 0.0409603114725260, -0.1577034342818964, -0.0186997671513728, 0.2516787797623318, 0.6377389692241489, 0.5445837000410291, 0.0025660396846260, -0.1278611173986804, -0.0080045221581074, 0.0303909779634732, 0.0010929716472247, -0.0054029061645258, 0.0003480830032355, 0.0006083566106577, -0.0000914924951015, -0.0000289109064442}}
                    };
                    auto it = coif_coeffs.find(N);
                    if (it != coif_coeffs.end())
                        return it->second;
                    XTENSOR_THROW(std::invalid_argument, "Coiflet of order " + std::to_string(N) + " not precomputed.");
                    return {};
                }

                // Biorthogonal spline wavelets (biorM.N)
                struct BiorFilterBank
                {
                    std::vector<double> dec_lo, dec_hi, rec_lo, rec_hi;
                };
                
                BiorFilterBank biorthogonal_filters(int M, int N)
                {
                    // Precomputed for common bior wavelets
                    static const std::map<std::pair<int,int>, BiorFilterBank> bior_coeffs = {
                        {{1,1}, { // bior1.1 (same as Haar)
                            {0.7071067811865475, 0.7071067811865475},
                            {-0.7071067811865475, 0.7071067811865475},
                            {0.7071067811865475, 0.7071067811865475},
                            {0.7071067811865475, -0.7071067811865475}
                        }},
                        {{1,3}, {
                            {-0.1767766952966369, 0.5303300858899107, 1.0606601717798214, 0.5303300858899107, -0.1767766952966369},
                            {0.0, 0.0, 0.3535533905932738, -0.7071067811865475, 0.3535533905932738},
                            {0.3535533905932738, 0.7071067811865475, 0.3535533905932738, 0.0, 0.0},
                            {0.1767766952966369, 0.5303300858899107, -1.0606601717798214, 0.5303300858899107, 0.1767766952966369}
                        }},
                        {{2,2}, {
                            {0.0, -0.1767766952966369, 0.5303300858899107, 0.5303300858899107, -0.1767766952966369, 0.0},
                            {0.0, 0.3535533905932738, -0.7071067811865475, 0.3535533905932738, 0.0, 0.0},
                            {0.0, 0.3535533905932738, 0.7071067811865475, 0.3535533905932738, 0.0, 0.0},
                            {0.0, 0.1767766952966369, 0.5303300858899107, -0.5303300858899107, -0.1767766952966369, 0.0}
                        }},
                        {{2,4}, {
                            {0.0, 0.0331456303681193, -0.0662912607362385, -0.1767766952966369, 0.4198446513295126, 0.9943689110435835, 0.4198446513295126, -0.1767766952966369, -0.0662912607362385, 0.0331456303681193},
                            {0.0, 0.0, 0.0, 0.3535533905932738, -0.7071067811865475, 0.3535533905932738, 0.0, 0.0, 0.0, 0.0},
                            {0.0, 0.0, 0.0, 0.3535533905932738, 0.7071067811865475, 0.3535533905932738, 0.0, 0.0, 0.0, 0.0},
                            {0.0, -0.0331456303681193, -0.0662912607362385, 0.1767766952966369, 0.4198446513295126, -0.9943689110435835, 0.4198446513295126, 0.1767766952966369, -0.0662912607362385, -0.0331456303681193}
                        }}
                    };
                    auto it = bior_coeffs.find({M,N});
                    if (it != bior_coeffs.end())
                        return it->second;
                    XTENSOR_THROW(std::invalid_argument, "Biorthogonal wavelet bior" + std::to_string(M) + "." + std::to_string(N) + " not precomputed.");
                    return {};
                }

                // Reverse biorthogonal filters (for adjoint)
                inline std::vector<double> reverse_bior(const std::vector<double>& f)
                {
                    return reverse_filter(f);
                }

            } // namespace detail

            // --------------------------------------------------------------------
            // Wavelet factory function
            // --------------------------------------------------------------------
            inline Wavelet create_wavelet(const std::string& name)
            {
                Wavelet w;
                w.name = name;

                if (name == "haar" || name == "db1")
                {
                    w.dec_lo = {0.7071067811865475, 0.7071067811865475};
                    w.dec_hi = detail::qmf(w.dec_lo);
                    w.rec_lo = w.dec_lo;
                    w.rec_hi = w.dec_hi;
                    w.vanishing_moments_psi = 1;
                    w.vanishing_moments_phi = 1;
                    w.orthogonal = true;
                    w.symmetric = true;
                }
                else if (name.rfind("db", 0) == 0)
                {
                    int order = std::stoi(name.substr(2));
                    w.dec_lo = detail::daubechies_scaling_filter(order);
                    w.dec_hi = detail::qmf(w.dec_lo);
                    w.rec_lo = detail::reverse_filter(w.dec_lo);
                    w.rec_hi = detail::reverse_filter(w.dec_hi);
                    w.vanishing_moments_psi = order;
                    w.orthogonal = true;
                }
                else if (name.rfind("sym", 0) == 0)
                {
                    int order = std::stoi(name.substr(3));
                    w.dec_lo = detail::symlets_scaling_filter(order);
                    w.dec_hi = detail::qmf(w.dec_lo);
                    w.rec_lo = detail::reverse_filter(w.dec_lo);
                    w.rec_hi = detail::reverse_filter(w.dec_hi);
                    w.vanishing_moments_psi = order;
                    w.orthogonal = true;
                    w.symmetric = true; // near symmetric
                }
                else if (name.rfind("coif", 0) == 0)
                {
                    int order = std::stoi(name.substr(4));
                    w.dec_lo = detail::coiflet_scaling_filter(order);
                    w.dec_hi = detail::qmf(w.dec_lo);
                    w.rec_lo = detail::reverse_filter(w.dec_lo);
                    w.rec_hi = detail::reverse_filter(w.dec_hi);
                    w.vanishing_moments_psi = 2 * order;
                    w.vanishing_moments_phi = 2 * order - 1;
                    w.orthogonal = true;
                }
                else if (name.rfind("bior", 0) == 0)
                {
                    // name like "bior1.3"
                    size_t dot = name.find('.');
                    int M = std::stoi(name.substr(4, dot - 4));
                    int N = std::stoi(name.substr(dot + 1));
                    auto fb = detail::biorthogonal_filters(M, N);
                    w.dec_lo = fb.dec_lo;
                    w.dec_hi = fb.dec_hi;
                    w.rec_lo = fb.rec_lo;
                    w.rec_hi = fb.rec_hi;
                    w.biorthogonal = true;
                    w.orthogonal = false;
                }
                else
                {
                    XTENSOR_THROW(std::invalid_argument, "Unknown wavelet name: " + name);
                }
                return w;
            }

            // --------------------------------------------------------------------
            // Single-level Discrete Wavelet Transform (DWT)
            // --------------------------------------------------------------------
            template <class E>
            inline auto dwt(const xexpression<E>& x, const Wavelet& wavelet, 
                            const std::string& mode = "periodization")
            {
                const auto& data = x.derived_cast();
                if (data.dimension() != 1)
                {
                    XTENSOR_THROW(std::invalid_argument, "dwt: input must be 1-D");
                }

                size_t n = data.size();
                std::vector<double> sig(n);
                for (size_t i = 0; i < n; ++i)
                    sig[i] = static_cast<double>(data(i));

                std::vector<double> cA, cD;
                if (mode == "periodization" || mode == "per")
                {
                    cA = detail::per_convolve_down(sig, wavelet.dec_lo);
                    cD = detail::per_convolve_down(sig, wavelet.dec_hi);
                }
                else if (mode == "zero" || mode == "zpd")
                {
                    cA = detail::convolve_down(sig, wavelet.dec_lo);
                    cD = detail::convolve_down(sig, wavelet.dec_hi);
                }
                else if (mode == "symmetric" || mode == "sym")
                {
                    // Symmetric extension (not fully implemented, fallback)
                    cA = detail::convolve_down(sig, wavelet.dec_lo);
                    cD = detail::convolve_down(sig, wavelet.dec_hi);
                }
                else
                {
                    XTENSOR_THROW(std::invalid_argument, "dwt: unsupported mode '" + mode + "'");
                }

                xarray_container<double> approx({cA.size()});
                xarray_container<double> detail({cD.size()});
                std::copy(cA.begin(), cA.end(), approx.begin());
                std::copy(cD.begin(), cD.end(), detail.begin());
                return std::make_pair(approx, detail);
            }

            // --------------------------------------------------------------------
            // Inverse Discrete Wavelet Transform (IDWT) - single level
            // --------------------------------------------------------------------
            template <class E1, class E2>
            inline auto idwt(const xexpression<E1>& cA, const xexpression<E2>& cD,
                             const Wavelet& wavelet, const std::string& mode = "periodization",
                             size_t target_len = 0)
            {
                const auto& approx = cA.derived_cast();
                const auto& detail = cD.derived_cast();

                if (approx.dimension() != 1 || detail.dimension() != 1)
                    XTENSOR_THROW(std::invalid_argument, "idwt: coefficients must be 1-D");

                size_t lenA = approx.size();
                size_t lenD = detail.size();
                if (lenA != lenD)
                    XTENSOR_THROW(std::invalid_argument, "idwt: approximation and detail lengths must match");

                std::vector<double> cA_vec(lenA), cD_vec(lenD);
                for (size_t i = 0; i < lenA; ++i) cA_vec[i] = static_cast<double>(approx(i));
                for (size_t i = 0; i < lenD; ++i) cD_vec[i] = static_cast<double>(detail(i));

                size_t L = target_len > 0 ? target_len : 2 * lenA;
                std::vector<double> rec;

                if (mode == "periodization" || mode == "per")
                {
                    auto rec_lo = detail::per_upsample_convolve(cA_vec, wavelet.rec_lo, L);
                    auto rec_hi = detail::per_upsample_convolve(cD_vec, wavelet.rec_hi, L);
                    rec.resize(L);
                    for (size_t i = 0; i < L; ++i)
                        rec[i] = rec_lo[i] + rec_hi[i];
                }
                else
                {
                    auto rec_lo = detail::upsample_convolve(cA_vec, wavelet.rec_lo, L);
                    auto rec_hi = detail::upsample_convolve(cD_vec, wavelet.rec_hi, L);
                    rec.resize(L);
                    for (size_t i = 0; i < L; ++i)
                        rec[i] = rec_lo[i] + rec_hi[i];
                }

                xarray_container<double> result({L});
                std::copy(rec.begin(), rec.end(), result.begin());
                return result;
            }

            // --------------------------------------------------------------------
            // Multi-level DWT (wavedec)
            // --------------------------------------------------------------------
            template <class E>
            inline auto wavedec(const xexpression<E>& x, const Wavelet& wavelet, int level,
                                const std::string& mode = "periodization")
            {
                if (level <= 0)
                {
                    XTENSOR_THROW(std::invalid_argument, "wavedec: level must be positive");
                }

                const auto& data = x.derived_cast();
                std::vector<xarray_container<double>> coeffs;
                coeffs.reserve(level + 1);

                xarray_container<double> current = eval(data);
                for (int l = 0; l < level; ++l)
                {
                    auto [cA, cD] = dwt(current, wavelet, mode);
                    coeffs.push_back(cD);
                    current = cA;
                    if (current.size() < wavelet.dec_lo.size())
                        break;
                }
                coeffs.push_back(current); // final approximation
                std::reverse(coeffs.begin(), coeffs.end()); // [cA_n, cD_n, cD_{n-1}, ..., cD_1]
                return coeffs;
            }

            // --------------------------------------------------------------------
            // Multi-level IDWT (waverec)
            // --------------------------------------------------------------------
            inline auto waverec(const std::vector<xarray_container<double>>& coeffs,
                                const Wavelet& wavelet, const std::string& mode = "periodization")
            {
                if (coeffs.empty())
                    XTENSOR_THROW(std::invalid_argument, "waverec: empty coefficients");

                // coeffs is [cA_n, cD_n, cD_{n-1}, ..., cD_1]
                xarray_container<double> rec = coeffs[0];
                for (size_t i = 1; i < coeffs.size(); ++i)
                {
                    rec = idwt(rec, coeffs[i], wavelet, mode);
                }
                return rec;
            }

            // --------------------------------------------------------------------
            // Wavelet Packet Decomposition (wpdec)
            // --------------------------------------------------------------------
            inline std::map<std::string, xarray_container<double>> 
            wpdec(const xarray_container<double>& x, const Wavelet& wavelet, int level,
                  const std::string& mode = "periodization")
            {
                std::map<std::string, xarray_container<double>> tree;
                tree[""] = x;
                for (int l = 0; l < level; ++l)
                {
                    std::map<std::string, xarray_container<double>> new_level;
                    for (const auto& node : tree)
                    {
                        if (node.first.length() == static_cast<size_t>(l))
                        {
                            auto [cA, cD] = dwt(node.second, wavelet, mode);
                            new_level[node.first + "a"] = cA;
                            new_level[node.first + "d"] = cD;
                        }
                    }
                    tree.insert(new_level.begin(), new_level.end());
                }
                return tree;
            }

            // --------------------------------------------------------------------
            // Thresholding functions for denoising
            // --------------------------------------------------------------------
            enum class ThresholdMode { Soft, Hard, Garrote };

            template <class E>
            inline auto threshold(const xexpression<E>& coeffs, double thr, ThresholdMode mode = ThresholdMode::Soft)
            {
                auto result = eval(coeffs);
                for (auto& v : result)
                {
                    double val = static_cast<double>(v);
                    if (mode == ThresholdMode::Soft)
                    {
                        if (val > thr) v = val - thr;
                        else if (val < -thr) v = val + thr;
                        else v = 0;
                    }
                    else if (mode == ThresholdMode::Hard)
                    {
                        if (std::abs(val) < thr) v = 0;
                    }
                    else // Garrote
                    {
                        if (std::abs(val) < thr) v = 0;
                        else v = val - thr * thr / val;
                    }
                }
                return result;
            }

            // VisuShrink threshold (universal)
            inline double visushrink_threshold(const std::vector<xarray_container<double>>& coeffs, double sigma = -1.0)
            {
                // Use finest detail coefficients to estimate sigma
                if (coeffs.size() < 2) return 0.0;
                const auto& finest = coeffs.back();
                double median_abs = xt::median(xt::abs(finest))();
                if (sigma < 0)
                    sigma = median_abs / 0.6745; // robust estimator
                size_t n = finest.size();
                return sigma * std::sqrt(2.0 * std::log(static_cast<double>(n)));
            }

            // Denoise using wavelet thresholding
            template <class E>
            inline auto wdenoise(const xexpression<E>& x, const Wavelet& wavelet, int level = 0,
                                 const std::string& mode = "periodization",
                                 ThresholdMode thresh_mode = ThresholdMode::Soft,
                                 const std::string& thresh_method = "visushrink")
            {
                const auto& data = x.derived_cast();
                if (level == 0)
                {
                    level = static_cast<int>(std::floor(std::log2(data.size()))) - 2;
                    level = std::max(1, level);
                }
                auto coeffs = wavedec(data, wavelet, level, mode);
                double thr = 0.0;
                if (thresh_method == "visushrink")
                    thr = visushrink_threshold(coeffs);
                else if (thresh_method == "sqtwolog")
                    thr = visushrink_threshold(coeffs); // same formula
                else
                    XTENSOR_THROW(std::invalid_argument, "Unknown threshold method: " + thresh_method);

                // Apply threshold to all detail coefficients (skip approximation)
                for (size_t i = 1; i < coeffs.size(); ++i)
                {
                    coeffs[i] = threshold(coeffs[i], thr, thresh_mode);
                }
                return waverec(coeffs, wavelet, mode);
            }

            // --------------------------------------------------------------------
            // Stationary Wavelet Transform (SWT) / undecimated
            // --------------------------------------------------------------------
            template <class E>
            inline auto swt(const xexpression<E>& x, const Wavelet& wavelet, int level)
            {
                // Not fully implemented, placeholder that returns wavedec (decimated)
                return wavedec(x, wavelet, level, "periodization");
            }

            // --------------------------------------------------------------------
            // Wavelet packet best basis selection (using entropy)
            // --------------------------------------------------------------------
            inline double shannon_entropy(const xarray_container<double>& x)
            {
                double sum_sq = xt::sum(x * x)();
                if (sum_sq == 0) return 0;
                auto p = (x * x) / sum_sq;
                double ent = 0;
                for (auto v : p)
                    if (v > 0) ent -= v * std::log(v);
                return ent;
            }

            inline std::string best_basis(const std::map<std::string, xarray_container<double>>& wp_tree,
                                          const std::function<double(const xarray_container<double>&)>& cost_func = shannon_entropy)
            {
                // Bottom-up pruning to find best basis (simplified)
                // Returns the node name with minimal cost
                double best_cost = std::numeric_limits<double>::max();
                std::string best_node;
                for (const auto& node : wp_tree)
                {
                    double cost = cost_func(node.second);
                    if (cost < best_cost)
                    {
                        best_cost = cost;
                        best_node = node.first;
                    }
                }
                return best_node;
            }

            // --------------------------------------------------------------------
            // Continuous Wavelet Transform (CWT) using convolution
            // --------------------------------------------------------------------
            template <class E>
            inline auto cwt(const xexpression<E>& x, const Wavelet& wavelet, 
                            const std::vector<double>& scales)
            {
                const auto& signal = x.derived_cast();
                if (signal.dimension() != 1)
                    XTENSOR_THROW(std::invalid_argument, "cwt: input must be 1-D");

                size_t n = signal.size();
                xarray_container<double> result({scales.size(), n});

                // For each scale, compute convolution with scaled wavelet
                for (size_t s_idx = 0; s_idx < scales.size(); ++s_idx)
                {
                    double scale = scales[s_idx];
                    // Build scaled wavelet filter (approximate continuous wavelet via sampling)
                    // Use the reconstruction low-pass as mother wavelet approximation
                    std::vector<double> mother = wavelet.dec_lo; // placeholder; real CWT would use wavelet function
                    // Scale and sample
                    size_t filter_len = static_cast<size_t>(std::ceil(10.0 * scale));
                    if (filter_len < 4) filter_len = 4;
                    std::vector<double> scaled_wavelet(filter_len);
                    for (size_t i = 0; i < filter_len; ++i)
                    {
                        double t = (static_cast<double>(i) - filter_len/2.0) / scale;
                        // Simple Mexican hat as example; for real wavelets, need proper function
                        scaled_wavelet[i] = (1.0 - t*t) * std::exp(-t*t/2.0);
                    }
                    // Convolve
                    auto conv = convolve(signal, scaled_wavelet, "same");
                    for (size_t i = 0; i < n; ++i)
                        result(s_idx, i) = conv(i);
                }
                return result;
            }

            // --------------------------------------------------------------------
            // Scalogram (power of CWT)
            // --------------------------------------------------------------------
            template <class E>
            inline auto scalogram(const xexpression<E>& x, const Wavelet& wavelet,
                                  const std::vector<double>& scales)
            {
                auto cwt_mat = cwt(x, wavelet, scales);
                return xt::abs(cwt_mat) * xt::abs(cwt_mat);
            }

            // --------------------------------------------------------------------
            // Utilities: max decomposition level
            // --------------------------------------------------------------------
            inline int wmaxlev(size_t n, const Wavelet& wavelet)
            {
                int lev = 0;
                size_t len = n;
                while (len >= wavelet.dec_lo.size())
                {
                    len = (len + 1) / 2;
                    ++lev;
                }
                return std::max(0, lev - 1);
            }

            // --------------------------------------------------------------------
            // Pad signal to dyadic length
            // --------------------------------------------------------------------
            template <class E>
            inline auto dyadup(const xexpression<E>& x, int level = 0)
            {
                auto data = eval(x);
                if (level <= 0)
                    level = static_cast<int>(std::ceil(std::log2(data.size())));
                size_t target_len = static_cast<size_t>(1) << level;
                if (data.size() == target_len) return data;
                xarray_container<double> padded({target_len}, 0.0);
                std::copy(data.begin(), data.end(), padded.begin());
                return padded;
            }

            // --------------------------------------------------------------------
            // Extract approximation coefficients at specific level
            // --------------------------------------------------------------------
            inline auto appcoef(const std::vector<xarray_container<double>>& coeffs, int level)
            {
                // coeffs is [cA_n, cD_n, ..., cD_1]
                // level 1 = cD_1, level N = cD_N, approximation is at index 0
                if (level < 0 || static_cast<size_t>(level) >= coeffs.size())
                    XTENSOR_THROW(std::out_of_range, "appcoef: level out of range");
                return coeffs[0]; // approximation coefficients are always first
            }

            inline auto detcoef(const std::vector<xarray_container<double>>& coeffs, int level)
            {
                if (level <= 0 || static_cast<size_t>(level) >= coeffs.size())
                    XTENSOR_THROW(std::out_of_range, "detcoef: level out of range");
                return coeffs[coeffs.size() - static_cast<size_t>(level)];
            }

        } // namespace wavelet

        // Bring wavelet functions into xt namespace
        using wavelet::Wavelet;
        using wavelet::create_wavelet;
        using wavelet::dwt;
        using wavelet::idwt;
        using wavelet::wavedec;
        using wavelet::waverec;
        using wavelet::wpdec;
        using wavelet::ThresholdMode;
        using wavelet::threshold;
        using wavelet::visushrink_threshold;
        using wavelet::wdenoise;
        using wavelet::swt;
        using wavelet::cwt;
        using wavelet::scalogram;
        using wavelet::wmaxlev;
        using wavelet::dyadup;
        using wavelet::appcoef;
        using wavelet::detcoef;
        using wavelet::shannon_entropy;
        using wavelet::best_basis;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XWAVELET_HPP

// signal/xwavelet.hpp
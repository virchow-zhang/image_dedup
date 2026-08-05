// dedup_core.exe — 科研图片查重原生内核 (C++17 + OpenCV)
//
// 子命令:
//   hash     --dir <目录> --out <hashes.bin>           批量感知哈希
//   mih      --in <hashes.bin> --threshold N --out <pairs.txt>  MIH 候选对
//   cmfd     --image <路径> [--out <regions.json>]     同图内部复制检测
//   template --big <大图> --small <小图> [--out <res.json>]  多尺度模板匹配
//
// 输出文本一律 UTF-8。
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/geometry/2d.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// OpenCV 5: ROTATE_* 枚举移除, 使用整型常量
#define ROTATE_90_CW 0
#define ROTATE_180 1
#define ROTATE_90_CCW 2

using std::string;
using std::vector;

// ---------------------------------------------------------------- 工具

static string json_escape(const string &s) {
    std::ostringstream o;
    for (char c : s) {
        if (c == '"') o << "\\\"";
        else if (c == '\\') o << "\\\\";
        else if (c == '\n') o << "\\n";
        else if (c == '\r') o << "\\r";
        else if (c == '\t') o << "\\t";
        else if (static_cast<unsigned char>(c) < 0x20) {
            char buf[8];
            snprintf(buf, sizeof buf, "\\u%04x", c);
            o << buf;
        } else o << c;
    }
    return o.str();
}

static cv::Mat read_gray(const string &path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) return img;
    if (img.depth() != CV_8U) img.convertTo(img, CV_8U, 255.0 / 65535.0);
    return img;
}

// ---------------------------------------------------------------- 感知哈希

// 32 点一维 DCT-II
static void dct_1d(const double in[32], double out[32]) {
    for (int k = 0; k < 32; ++k) {
        double s = 0.0;
        for (int n = 0; n < 32; ++n) s += in[n] * std::cos(M_PI * k * (2.0 * n + 1.0) / 64.0);
        out[k] = (k == 0 ? std::sqrt(1.0 / 32.0) : std::sqrt(2.0 / 32.0)) * s;
    }
}

// 可分离二维 DCT
static void dct_2d(const cv::Mat &src, cv::Mat &dst) {
    const int N = 32;
    cv::Mat rows(N, N, CV_64F);
    for (int y = 0; y < N; ++y) {
        double in[32], out[32];
        for (int x = 0; x < N; ++x) in[x] = src.at<double>(y, x);
        dct_1d(in, out);
        for (int x = 0; x < N; ++x) rows.at<double>(y, x) = out[x];
    }
    dst = cv::Mat(N, N, CV_64F);
    for (int x = 0; x < N; ++x) {
        double in[32], out[32];
        for (int y = 0; y < N; ++y) in[y] = rows.at<double>(y, x);
        dct_1d(in, out);
        for (int y = 0; y < N; ++y) dst.at<double>(y, x) = out[y];
    }
}

// pHash: 32x32 灰度 -> 16x16 DCT 低频符号 -> 256 bit
static uint64_t phash_256(const cv::Mat &gray, uint64_t out[4]) {
    cv::Mat small;
    cv::resize(gray, small, cv::Size(32, 32), 0, 0, cv::INTER_AREA);
    small.convertTo(small, CV_64F);
    cv::Mat dct;
    dct_2d(small, dct);
    cv::Mat low = dct(cv::Rect(0, 0, 16, 16));
    double mean = cv::mean(low)[0];
    for (int i = 0; i < 4; ++i) out[i] = 0;
    for (int y = 0; y < 16; ++y)
        for (int x = 0; x < 16; ++x) {
            int bit = y * 16 + x;
            if (low.at<double>(y, x) > mean) out[bit / 64] |= (uint64_t(1) << (bit % 64));
        }
    return 0;
}

// dHash: 17x17 -> 16x16 相邻比较 -> 256 bit
static uint64_t dhash_256(const cv::Mat &gray, uint64_t out[4]) {
    cv::Mat small;
    cv::resize(gray, small, cv::Size(17, 17), 0, 0, cv::INTER_AREA);
    for (int i = 0; i < 4; ++i) out[i] = 0;
    for (int y = 0; y < 16; ++y)
        for (int x = 0; x < 16; ++x) {
            int bit = y * 16 + x;
            if (small.at<uchar>(y, x) > small.at<uchar>(y, x + 1))
                out[bit / 64] |= (uint64_t(1) << (bit % 64));
        }
    return 0;
}

// aHash: 16x16 与均值比较 -> 256 bit
static uint64_t ahash_256(const cv::Mat &gray, uint64_t out[4]) {
    cv::Mat small;
    cv::resize(gray, small, cv::Size(16, 16), 0, 0, cv::INTER_AREA);
    double mean = cv::mean(small)[0];
    for (int i = 0; i < 4; ++i) out[i] = 0;
    for (int y = 0; y < 16; ++y)
        for (int x = 0; x < 16; ++x) {
            int bit = y * 16 + x;
            if (small.at<uchar>(y, x) > mean)
                out[bit / 64] |= (uint64_t(1) << (bit % 64));
        }
    return 0;
}

static int popcount64(uint64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return static_cast<int>((x * 0x0101010101010101ULL) >> 56);
}

static int hamming_256(const uint64_t a[4], const uint64_t b[4]) {
    return popcount64(a[0] ^ b[0]) + popcount64(a[1] ^ b[1]) +
           popcount64(a[2] ^ b[2]) + popcount64(a[3] ^ b[3]);
}

// ---------------------------------------------------------------- 哈希文件

struct HashRecord {
    string path;
    uint64_t phash[4];
    uint64_t dhash[4];
    uint64_t ahash[4];
    uint64_t variants[7][4];  // rot90 rot180 rot270 flipH flipV rot15 rot345
};

static void write_hashes(const string &out_path, const vector<HashRecord> &recs) {
    std::ofstream f(out_path, std::ios::binary);
    f.write("DDHB", 4);
    uint32_t version = 1;
    f.write(reinterpret_cast<const char *>(&version), 4);
    uint64_t n = recs.size();
    f.write(reinterpret_cast<const char *>(&n), 8);
    for (const auto &r : recs) {
        uint32_t plen = static_cast<uint32_t>(r.path.size());
        f.write(reinterpret_cast<const char *>(&plen), 4);
        f.write(r.path.data(), plen);
        f.write(reinterpret_cast<const char *>(r.phash), 32);
        f.write(reinterpret_cast<const char *>(r.dhash), 32);
        f.write(reinterpret_cast<const char *>(r.ahash), 32);
        f.write(reinterpret_cast<const char *>(r.variants), 224);
    }
}

static bool read_hashes(const string &in_path, vector<HashRecord> &recs) {
    std::ifstream f(in_path, std::ios::binary);
    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "DDHB", 4) != 0) return false;
    uint32_t version;
    f.read(reinterpret_cast<char *>(&version), 4);
    uint64_t n;
    f.read(reinterpret_cast<char *>(&n), 8);
    recs.resize(n);
    for (auto &r : recs) {
        uint32_t plen;
        f.read(reinterpret_cast<char *>(&plen), 4);
        r.path.resize(plen);
        f.read(&r.path[0], plen);
        f.read(reinterpret_cast<char *>(r.phash), 32);
        f.read(reinterpret_cast<char *>(r.dhash), 32);
        f.read(reinterpret_cast<char *>(r.ahash), 32);
        f.read(reinterpret_cast<char *>(r.variants), 224);
    }
    return true;
}

// ---------------------------------------------------------------- hash 子命令

static vector<string> list_images(const string &dir) {
    vector<string> out;
    const vector<string> exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"};
    for (const auto &entry : std::filesystem::recursive_directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        string p = entry.path().string();
        string ext = p.substr(p.find_last_of('.'));
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (std::find(exts.begin(), exts.end(), ext) != exts.end())
            out.push_back(p);
    }
    std::sort(out.begin(), out.end());
    return out;
}

static int cmd_hash(const vector<string> &args) {
    string dir, out_path;
    for (size_t i = 0; i + 1 < args.size(); i += 2) {
        if (args[i] == "--dir") dir = args[i + 1];
        else if (args[i] == "--out") out_path = args[i + 1];
    }
    if (dir.empty() || out_path.empty()) {
        std::cerr << "usage: dedup_core hash --dir <dir> --out <hashes.bin>\n";
        return 2;
    }
    auto files = list_images(dir);
    std::cerr << "found " << files.size() << " images\n";
    vector<HashRecord> recs;
    recs.reserve(files.size());
    for (size_t i = 0; i < files.size(); ++i) {
        cv::Mat gray = read_gray(files[i]);
        if (gray.empty()) {
            std::cerr << "  [skip] " << files[i] << "\n";
            continue;
        }
        HashRecord r;
        r.path = files[i];
        phash_256(gray, r.phash);
        dhash_256(gray, r.dhash);
        ahash_256(gray, r.ahash);
        // 90/180/270 旋转 (保持原始 gray 不变, 用副本)
        cv::Mat r1, r2, r3;
        cv::rotate(gray, r1, ROTATE_90_CW);
        cv::rotate(gray, r2, ROTATE_180);
        cv::rotate(gray, r3, ROTATE_90_CCW);
        phash_256(r1, r.variants[0]);
        phash_256(r2, r.variants[1]);
        phash_256(r3, r.variants[2]);
        cv::Mat fh, fv;
        cv::flip(gray, fh, 1);
        cv::flip(gray, fv, 0);
        phash_256(fh, r.variants[3]);
        phash_256(fv, r.variants[4]);
        // 自由旋转 15° / -15° (复制边框)
        auto rotate_free = [&](double ang, uint64_t dst[4]) {
            int h = gray.rows, w = gray.cols;
            double rad = ang * M_PI / 180.0;
            int nw = static_cast<int>(std::abs(w * std::cos(rad)) + std::abs(h * std::sin(rad))) + 1;
            int nh = static_cast<int>(std::abs(w * std::sin(rad)) + std::abs(h * std::cos(rad))) + 1;
            cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(w / 2.0f, h / 2.0f), ang, 1.0);
            M.at<double>(0, 2) += (nw - w) / 2.0;
            M.at<double>(1, 2) += (nh - h) / 2.0;
            cv::Mat out;
            cv::warpAffine(gray, out, M, cv::Size(nw, nh), cv::INTER_LINEAR,
                           cv::BORDER_REPLICATE);
            phash_256(out, dst);
        };
        rotate_free(15, r.variants[5]);
        rotate_free(-15, r.variants[6]);
        recs.push_back(std::move(r));
        if ((i + 1) % 100 == 0) std::cerr << "  hashed " << (i + 1) << "/" << files.size() << "\n";
    }
    write_hashes(out_path, recs);
    std::cerr << "wrote " << recs.size() << " records -> " << out_path << "\n";
    return 0;
}

// ---------------------------------------------------------------- MIH

static int hamming_of(const HashRecord &a, const HashRecord &b, int src) {
    // src: 0=phash 1=dhash 2=ahash 3..9=variants[src-3]
    if (src == 0) return hamming_256(a.phash, b.phash);
    if (src == 1) return hamming_256(a.dhash, b.dhash);
    if (src == 2) return hamming_256(a.ahash, b.ahash);
    return hamming_256(a.variants[src - 3], b.variants[src - 3]);
}

static void mih_source(const vector<HashRecord> &recs, int src, int threshold,
                       vector<std::pair<int, int>> &out) {
    const int n = static_cast<int>(recs.size());
    const int nt = 16;
    const bool is_variant = src >= 3;
    // 行布局: 非变体源 rows = n (全是源哈希);
    // 变体源 rows = 2n: 前 n 行 = 变体哈希, 后 n 行 = plain phash,
    // 命中即 phash(变体(A)) vs phash(B)
    const int rows = is_variant ? 2 * n : n;
    auto bits_of = [&](int row, uint64_t dst[4]) {
        int rec = row % n;
        if (is_variant) {
            std::memcpy(dst, row < n ? recs[rec].variants[src - 3]
                                     : recs[rec].phash, 32);
        } else if (src == 0) {
            std::memcpy(dst, recs[rec].phash, 32);
        } else if (src == 1) {
            std::memcpy(dst, recs[rec].dhash, 32);
        } else {
            std::memcpy(dst, recs[rec].ahash, 32);
        }
    };
    vector<vector<uint16_t>> keys(nt, vector<uint16_t>(rows));
    for (int r = 0; r < rows; ++r) {
        uint64_t bits[4];
        bits_of(r, bits);
        for (int t = 0; t < nt; ++t) {
            uint16_t key = 0;
            for (int j = 0; j < 16; ++j) {
                int bit = t + j * 16;
                if ((bits[bit / 64] >> (bit % 64)) & 1) key |= (1 << j);
            }
            keys[t][r] = key;
        }
    }
    std::set<std::pair<int, int>> seen;
    for (int t = 0; t < nt; ++t) {
        vector<int> order(rows);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return keys[t][a] < keys[t][b]; });
        size_t i = 0;
        while (i < (size_t)rows) {
            size_t j = i + 1;
            while (j < (size_t)rows && keys[t][order[j]] == keys[t][order[i]]) ++j;
            if (j - i >= 2) {
                for (size_t a = i; a < j; ++a)
                    for (size_t b = a + 1; b < j; ++b) {
                        int x = order[a], y = order[b];
                        if (x == y) continue;
                        if (x % n == y % n) continue;  // 同一张图
                        uint64_t bx[4], by[4];
                        bits_of(x, bx);
                        bits_of(y, by);
                        if (hamming_256(bx, by) <= threshold)
                            seen.insert(x % n < y % n
                                            ? std::make_pair(x % n, y % n)
                                            : std::make_pair(y % n, x % n));
                    }
            }
            i = j;
        }
    }
    for (auto &p : seen) out.push_back(p);
}

static int cmd_mih(const vector<string> &args) {
    string in_path, out_path;
    int threshold = 5;
    for (size_t i = 0; i + 1 < args.size(); i += 2) {
        if (args[i] == "--in") in_path = args[i + 1];
        else if (args[i] == "--out") out_path = args[i + 1];
        else if (args[i] == "--threshold") threshold = std::stoi(args[i + 1]);
    }
    if (in_path.empty() || out_path.empty() || threshold >= 16) {
        std::cerr << "usage: dedup_core mih --in <hashes.bin> --threshold N --out <pairs.txt>\n";
        return 2;
    }
    vector<HashRecord> recs;
    if (!read_hashes(in_path, recs)) {
        std::cerr << "cannot read " << in_path << "\n";
        return 1;
    }
    std::set<std::pair<int, int>> all;
    vector<std::pair<int, int>> tmp;
    for (int src = 0; src < 10; ++src) {
        tmp.clear();
        mih_source(recs, src, threshold, tmp);
        all.insert(tmp.begin(), tmp.end());
        std::cerr << "  src " << src << ": " << tmp.size() << " pairs\n";
    }
    std::ofstream out(out_path);
    for (auto &p : all) out << recs[p.first].path << "\t" << recs[p.second].path << "\n";
    std::cerr << "mih total: " << all.size() << " pairs\n";
    return 0;
}

// ---------------------------------------------------------------- cmfd

static int cmd_cmfd(const vector<string> &args) {
    string image, out_path;
    int min_matches = 6;
    for (size_t i = 0; i + 1 < args.size(); i += 2) {
        if (args[i] == "--image") image = args[i + 1];
        else if (args[i] == "--out") out_path = args[i + 1];
        else if (args[i] == "--min-matches") min_matches = std::stoi(args[i + 1]);
    }
    if (image.empty()) {
        std::cerr << "usage: dedup_core cmfd --image <path> [--out res.json]\n";
        return 2;
    }
    cv::Mat gray = read_gray(image);
    if (gray.empty()) return 1;
    if (std::max(gray.rows, gray.cols) > 2048) {
        double s = 2048.0 / std::max(gray.rows, gray.cols);
        cv::resize(gray, gray, cv::Size(0, 0), s, s, cv::INTER_AREA);
    }
    auto sift = cv::SIFT::create(3000, 3, 0.04, 10, 1.6);
    std::vector<cv::KeyPoint> kp;
    cv::Mat des;
    sift->detectAndCompute(gray, cv::noArray(), kp, des);
    std::ostringstream out;
    out << "{\"keypoints\":" << kp.size() << ",\"regions\":[";
    if (des.rows >= min_matches * 2) {
        cv::BFMatcher bf(cv::NORM_L2);
        std::vector<std::vector<cv::DMatch>> knn;
        bf.knnMatch(des, des, knn, 3);
        struct Clu {
            std::vector<int> idx;
            double mx = 0, my = 0;
        };
        std::map<std::pair<int, int>, Clu> clusters;
        for (auto &trio : knn) {
            cv::DMatch *m = nullptr, *n = nullptr;
            for (auto &mm : trio) {
                if (mm.queryIdx == mm.trainIdx) continue;
                if (!m) m = &mm;
                else if (!n) n = &mm;
                else break;
            }
            if (!m || !n) continue;
            if (m->distance < 0.8 * n->distance) {
                const auto &p = kp[m->queryIdx].pt;
                const auto &q = kp[m->trainIdx].pt;
                int dx = static_cast<int>(std::lround((q.x - p.x) / 32.0));
                int dy = static_cast<int>(std::lround((q.y - p.y) / 32.0));
                clusters[{dx, dy}].idx.push_back(static_cast<int>(m->queryIdx));
            }
        }
        bool first = true;
        int ow = gray.cols, oh = gray.rows;
        for (auto &[key, clu] : clusters) {
            if ((int)clu.idx.size() < min_matches) continue;
            if (std::abs(key.first * 32.0) + std::abs(key.second * 32.0) < 64.0) continue;
            double mx = 0, my = 0;
            double x0 = 1e18, y0 = 1e18, x1 = -1e18, y1 = -1e18;
            for (int k : clu.idx) {
                const auto &p = kp[k].pt;
                mx += p.x; my += p.y;
                x0 = std::min(x0, (double)p.x); y0 = std::min(y0, (double)p.y);
                x1 = std::max(x1, (double)p.x); y1 = std::max(y1, (double)p.y);
            }
            mx /= clu.idx.size(); my /= clu.idx.size();
            if (!first) out << ",";
            first = false;
            out << "{\"n\":" << clu.idx.size()
                << ",\"dx\":" << (int)std::lround(mx) << ",\"dy\":" << (int)std::lround(my)
                << ",\"src\":[" << (int)x0 << "," << (int)y0 << ","
                << (int)(x1 - x0) << "," << (int)(y1 - y0) << "]}";
        }
    }
    out << "]}";
    if (!out_path.empty()) {
        std::ofstream f(out_path);
        f << out.str();
    } else {
        std::cout << out.str() << "\n";
    }
    return 0;
}

// ---------------------------------------------------------------- template

static int cmd_template(const vector<string> &args) {
    string big_path, small_path, out_path;
    for (size_t i = 0; i + 1 < args.size(); i += 2) {
        if (args[i] == "--big") big_path = args[i + 1];
        else if (args[i] == "--small") small_path = args[i + 1];
        else if (args[i] == "--out") out_path = args[i + 1];
    }
    if (big_path.empty() || small_path.empty()) {
        std::cerr << "usage: dedup_core template --big <大图> --small <小图> [--out res.json]\n";
        return 2;
    }
    cv::Mat big = read_gray(big_path);
    cv::Mat small = read_gray(small_path);
    if (big.empty() || small.empty()) return 1;
    if (std::max(big.rows, big.cols) > 1024) {
        double s = 1024.0 / std::max(big.rows, big.cols);
        cv::resize(big, big, cv::Size(0, 0), s, s, cv::INTER_AREA);
    }
    double best = -1;
    int bx = 0, by = 0;
    for (double sc : {1.0, 0.85, 0.7, 0.55}) {
        int tw = static_cast<int>(small.cols * sc);
        int th = static_cast<int>(small.rows * sc);
        if (tw < 24 || th < 24 || tw >= big.cols || th >= big.rows) continue;
        cv::Mat tmpl;
        cv::resize(small, tmpl, cv::Size(tw, th), 0, 0, cv::INTER_AREA);
        cv::Mat res;
        cv::matchTemplate(big, tmpl, res, cv::TM_CCOEFF_NORMED);
        double m;
        cv::Point loc;
        cv::minMaxLoc(res, nullptr, &m, nullptr, &loc);
        if (m > best) {
            best = m;
            bx = loc.x;
            by = loc.y;
        }
    }
    std::ostringstream o;
    o << "{\"score\":" << best << ",\"x\":" << bx << ",\"y\":" << by << "}";
    if (!out_path.empty()) {
        std::ofstream f(out_path);
        f << o.str();
    } else {
        std::cout << o.str() << "\n";
    }
    return 0;
}

// ---------------------------------------------------------------- main

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "dedup_core: hash|mih|cmfd|template\n";
        return 2;
    }
    vector<string> args(argv + 2, argv + argc);
    string cmd = argv[1];
    try {
        if (cmd == "hash") return cmd_hash(args);
        if (cmd == "mih") return cmd_mih(args);
        if (cmd == "cmfd") return cmd_cmfd(args);
        if (cmd == "template") return cmd_template(args);
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    std::cerr << "unknown command: " << cmd << "\n";
    return 2;
}

#include "obj_loader.h"

#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

std::vector<std::string> loadLines(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open OBJ file: " + path);
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    return lines;
}

static std::vector<std::string> splitWhitespace(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream iss(s);
    std::string tok;
    while (iss >> tok) {
        out.push_back(tok);
    }
    return out;
}

static int parseObjIndexToken(const std::string& tok, int vertexCount) {
    const size_t slash = tok.find('/');
    const std::string s = (slash == std::string::npos) ? tok : tok.substr(0, slash);

    if (s.empty()) return -1;

    int idx = 0;
    try {
        idx = std::stoi(s);
    }
    catch (...) {
        return -1;
    }

    if (idx < 0) idx = vertexCount + idx;
    else idx = idx - 1;

    return idx;
}

ObjMesh parseOBJ(
    const std::vector<std::string>& lines,
    float modelScale,
    float tx,
    float ty,
    float tz
) {
    std::vector<float> verts;
    std::vector<uint32_t> tris;
    std::vector<uint32_t> triMaterials;

    std::unordered_map<std::string, uint32_t> materialMap;
    uint32_t currentMat = 0;
    uint32_t nextMatId = 0;

    for (const std::string& raw : lines) {
        std::string line = raw;

        const size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;
        const size_t last = line.find_last_not_of(" \t\r\n");
        line = line.substr(first, last - first + 1);

        if (line.empty() || line[0] == '#') continue;

        const std::vector<std::string> parts = splitWhitespace(line);
        if (parts.empty()) continue;

        if (parts[0] == "usemtl") {
            if (parts.size() > 1) {
                const std::string& name = parts[1];
                auto it = materialMap.find(name);
                if (it == materialMap.end()) {
                    materialMap[name] = nextMatId;
                    currentMat = nextMatId;
                    ++nextMatId;
                }
                else {
                    currentMat = it->second;
                }
            }
        }
        else if (parts[0] == "v") {
            if (parts.size() < 4) continue;

            const float x = std::stof(parts[1]) * modelScale + tx;
            const float y = std::stof(parts[2]) * modelScale + ty;
            const float z = std::stof(parts[3]) * modelScale + tz;

            verts.push_back(x);
            verts.push_back(y);
            verts.push_back(z);
        }
        else if (parts[0] == "f") {
            const int vertexCount = static_cast<int>(verts.size() / 3);

            std::vector<int> idx;
            idx.reserve(parts.size() - 1);

            for (size_t i = 1; i < parts.size(); ++i) {
                const int parsed = parseObjIndexToken(parts[i], vertexCount);
                if (parsed >= 0) {
                    idx.push_back(parsed);
                }
            }

            if (idx.size() < 3) continue;

            for (size_t i = 1; i + 1 < idx.size(); ++i) {
                tris.push_back(static_cast<uint32_t>(idx[0]));
                tris.push_back(static_cast<uint32_t>(idx[i]));
                tris.push_back(static_cast<uint32_t>(idx[i + 1]));
                triMaterials.push_back(currentMat);
            }
        }
    }

    ObjMesh mesh;
    mesh.vertices = std::move(verts);
    mesh.triangles = std::move(tris);
    mesh.triangleMaterials = std::move(triMaterials);
    return mesh;
}

void rotateModelAroundCenter(
    std::vector<float>& vertices,
    float pitch,
    float yaw,
    float roll
) {
    float minX = std::numeric_limits<float>::infinity();
    float minY = std::numeric_limits<float>::infinity();
    float minZ = std::numeric_limits<float>::infinity();
    float maxX = -std::numeric_limits<float>::infinity();
    float maxY = -std::numeric_limits<float>::infinity();
    float maxZ = -std::numeric_limits<float>::infinity();

    for (size_t i = 0; i + 2 < vertices.size(); i += 3) {
        const float x = vertices[i + 0];
        const float y = vertices[i + 1];
        const float z = vertices[i + 2];

        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (z < minZ) minZ = z;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
        if (z > maxZ) maxZ = z;
    }

    const float cx = 0.5f * (minX + maxX);
    const float cy = 0.5f * (minY + maxY);
    const float cz = 0.5f * (minZ + maxZ);

    const float cp = std::cos(pitch), sp = std::sin(pitch);
    const float cyw = std::cos(yaw), syw = std::sin(yaw);
    const float cr = std::cos(roll), sr = std::sin(roll);

    for (size_t i = 0; i + 2 < vertices.size(); i += 3) {
        float x = vertices[i + 0] - cx;
        float y = vertices[i + 1] - cy;
        float z = vertices[i + 2] - cz;

        float x1 = cyw * x + syw * z;
        float y1 = y;
        float z1 = -syw * x + cyw * z;

        float x2 = x1;
        float y2 = cp * y1 - sp * z1;
        float z2 = sp * y1 + cp * z1;

        float x3 = cr * x2 - sr * y2;
        float y3 = sr * x2 + cr * y2;
        float z3 = z2;

        vertices[i + 0] = x3 + cx;
        vertices[i + 1] = y3 + cy;
        vertices[i + 2] = z3 + cz;
    }
}
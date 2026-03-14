#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct ObjMesh {
    std::vector<float> vertices;
    std::vector<uint32_t> triangles;
    std::vector<uint32_t> triangleMaterials;
};

std::vector<std::string> loadLines(const std::string& path);

ObjMesh parseOBJ(
    const std::vector<std::string>& lines,
    float modelScale,
    float tx,
    float ty,
    float tz
);

void rotateModelAroundCenter(
    std::vector<float>& vertices,
    float pitch,
    float yaw,
    float roll
);
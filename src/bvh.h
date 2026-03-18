#pragma once

#include <cstdint>
#include <vector>

struct Vertex {
    float x, y, z;
};

struct Aabb {
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
};

struct Material {
    float albedo_smoothness[4];
    float emissive_emissiveIntensity[4];
    float metallic_smoothShading_transmission_ior[4];
    float dispersion_thinFilmThickness_thinFilmIor_thinFilmSubstrateEta[4];
    float thinFilmSubstrateK_pad[4];
};

struct EmissiveTriangle {
    uint32_t triIndex;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;

    float area;
    float weight;
    float cdf;
    float pad3;
};

struct TransmissiveTriangle {
    uint32_t triIndex;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
    float area;
    float pmf;
    float cdf;
    float pad3;
};

struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    std::vector<uint32_t> triangleMaterialIndices;
    std::vector<Material> materials;
};

struct GpuBvhNode {
    float bmin[3];
    int leftChild;     // interior: left child index, leaf: -1

    float bmax[3];
    int rightChild;    // interior: right child index, leaf: -1

    int firstTri;      // leaf: offset into leafTriIndices, interior: 0
    int triCount;      // leaf: > 0, interior: 0
    int pad0;
    int pad1;
};

static_assert(sizeof(GpuBvhNode) == 48, "GpuBvhNode must be 48 bytes");

struct FlattenedBvh {
    std::vector<GpuBvhNode> nodes;
    std::vector<uint32_t> leafTriIndices;
    Aabb bounds;
};

FlattenedBvh buildBvh(const MeshData& mesh);
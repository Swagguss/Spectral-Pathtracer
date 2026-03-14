#include "bvh.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

    struct Aabb {
        float minX, minY, minZ;
        float maxX, maxY, maxZ;
    };

    struct TriRef {
        uint32_t triId;
        float cx, cy, cz;
        Aabb bounds;
    };

    struct BuildNode {
        Aabb bounds;
        bool leaf = false;

        int left = -1;
        int right = -1;

        uint32_t firstTri = 0;
        uint32_t triCount = 0;
    };

    constexpr uint32_t MAX_LEAF_TRIS = 4;

    Aabb emptyAabb() {
        return {
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity()
        };
    }

    void expandAabb(Aabb& b, float x, float y, float z) {
        b.minX = std::min(b.minX, x);
        b.minY = std::min(b.minY, y);
        b.minZ = std::min(b.minZ, z);
        b.maxX = std::max(b.maxX, x);
        b.maxY = std::max(b.maxY, y);
        b.maxZ = std::max(b.maxZ, z);
    }

    void expandAabb(Aabb& dst, const Aabb& src) {
        dst.minX = std::min(dst.minX, src.minX);
        dst.minY = std::min(dst.minY, src.minY);
        dst.minZ = std::min(dst.minZ, src.minZ);
        dst.maxX = std::max(dst.maxX, src.maxX);
        dst.maxY = std::max(dst.maxY, src.maxY);
        dst.maxZ = std::max(dst.maxZ, src.maxZ);
    }

    Aabb computeTriBounds(const MeshData& mesh, uint32_t triId) {
        const uint32_t i0 = mesh.indices[triId * 3 + 0];
        const uint32_t i1 = mesh.indices[triId * 3 + 1];
        const uint32_t i2 = mesh.indices[triId * 3 + 2];

        const Vertex& v0 = mesh.vertices[i0];
        const Vertex& v1 = mesh.vertices[i1];
        const Vertex& v2 = mesh.vertices[i2];

        Aabb b = emptyAabb();
        expandAabb(b, v0.x, v0.y, v0.z);
        expandAabb(b, v1.x, v1.y, v1.z);
        expandAabb(b, v2.x, v2.y, v2.z);
        return b;
    }

    TriRef buildTriRef(const MeshData& mesh, uint32_t triId) {
        const uint32_t i0 = mesh.indices[triId * 3 + 0];
        const uint32_t i1 = mesh.indices[triId * 3 + 1];
        const uint32_t i2 = mesh.indices[triId * 3 + 2];

        const Vertex& v0 = mesh.vertices[i0];
        const Vertex& v1 = mesh.vertices[i1];
        const Vertex& v2 = mesh.vertices[i2];

        TriRef t{};
        t.triId = triId;
        t.cx = (v0.x + v1.x + v2.x) / 3.0f;
        t.cy = (v0.y + v1.y + v2.y) / 3.0f;
        t.cz = (v0.z + v1.z + v2.z) / 3.0f;
        t.bounds = computeTriBounds(mesh, triId);
        return t;
    }

    Aabb computeBounds(const std::vector<TriRef>& tris, uint32_t begin, uint32_t end) {
        Aabb b = emptyAabb();
        for (uint32_t i = begin; i < end; ++i) {
            expandAabb(b, tris[i].bounds);
        }
        return b;
    }

    int chooseSplitAxis(const Aabb& b) {
        const float ex = b.maxX - b.minX;
        const float ey = b.maxY - b.minY;
        const float ez = b.maxZ - b.minZ;

        if (ey > ex && ey >= ez) return 1;
        if (ez > ex && ez >= ey) return 2;
        return 0;
    }

    float centroidOnAxis(const TriRef& t, int axis) {
        if (axis == 0) return t.cx;
        if (axis == 1) return t.cy;
        return t.cz;
    }

    int buildRecursive(
        std::vector<BuildNode>& nodes,
        std::vector<TriRef>& tris,
        uint32_t begin,
        uint32_t end
    ) {
        const uint32_t count = end - begin;
        if (count == 0) {
            throw std::runtime_error("buildRecursive got empty range");
        }

        BuildNode node{};
        node.bounds = computeBounds(tris, begin, end);

        const int nodeIndex = static_cast<int>(nodes.size());
        nodes.push_back(node);

        if (count <= MAX_LEAF_TRIS) {
            nodes[nodeIndex].leaf = true;
            nodes[nodeIndex].firstTri = begin;
            nodes[nodeIndex].triCount = count;
            return nodeIndex;
        }

        const int axis = chooseSplitAxis(node.bounds);
        const uint32_t mid = begin + count / 2;

        std::nth_element(
            tris.begin() + begin,
            tris.begin() + mid,
            tris.begin() + end,
            [axis](const TriRef& a, const TriRef& b) {
                return centroidOnAxis(a, axis) < centroidOnAxis(b, axis);
            }
        );

        if (mid == begin || mid == end) {
            nodes[nodeIndex].leaf = true;
            nodes[nodeIndex].firstTri = begin;
            nodes[nodeIndex].triCount = count;
            return nodeIndex;
        }

        const int left = buildRecursive(nodes, tris, begin, mid);
        const int right = buildRecursive(nodes, tris, mid, end);

        nodes[nodeIndex].leaf = false;
        nodes[nodeIndex].left = left;
        nodes[nodeIndex].right = right;
        nodes[nodeIndex].firstTri = 0;
        nodes[nodeIndex].triCount = 0;

        return nodeIndex;
    }

    int flattenRecursive(
        int buildNodeIndex,
        const std::vector<BuildNode>& buildNodes,
        const std::vector<TriRef>& sortedTris,
        FlattenedBvh& out
    ) {
        const BuildNode& src = buildNodes[buildNodeIndex];

        const int flatIndex = static_cast<int>(out.nodes.size());
        out.nodes.push_back({});

        out.nodes[flatIndex].bmin[0] = src.bounds.minX;
        out.nodes[flatIndex].bmin[1] = src.bounds.minY;
        out.nodes[flatIndex].bmin[2] = src.bounds.minZ;
        out.nodes[flatIndex].bmax[0] = src.bounds.maxX;
        out.nodes[flatIndex].bmax[1] = src.bounds.maxY;
        out.nodes[flatIndex].bmax[2] = src.bounds.maxZ;
        out.nodes[flatIndex].pad0 = 0;
        out.nodes[flatIndex].pad1 = 0;

        if (src.leaf) {
            out.nodes[flatIndex].leftChild = -1;
            out.nodes[flatIndex].rightChild = -1;
            out.nodes[flatIndex].firstTri = static_cast<int>(out.leafTriIndices.size());
            out.nodes[flatIndex].triCount = static_cast<int>(src.triCount);

            for (uint32_t i = 0; i < src.triCount; ++i) {
                out.leafTriIndices.push_back(sortedTris[src.firstTri + i].triId);
            }
        }
        else {
            const int leftFlat = flattenRecursive(src.left, buildNodes, sortedTris, out);
            const int rightFlat = flattenRecursive(src.right, buildNodes, sortedTris, out);

            out.nodes[flatIndex].leftChild = leftFlat;
            out.nodes[flatIndex].rightChild = rightFlat;
            out.nodes[flatIndex].firstTri = 0;
            out.nodes[flatIndex].triCount = 0;
        }

        return flatIndex;
    }

} // namespace

FlattenedBvh buildBvh(const MeshData& mesh) {
    if (mesh.vertices.empty()) {
        throw std::runtime_error("buildBvh: mesh has no vertices");
    }
    if (mesh.indices.empty() || (mesh.indices.size() % 3) != 0) {
        throw std::runtime_error("buildBvh: mesh index buffer is invalid");
    }

    const uint32_t triCount = static_cast<uint32_t>(mesh.indices.size() / 3);

    std::vector<TriRef> triRefs;
    triRefs.reserve(triCount);

    for (uint32_t triId = 0; triId < triCount; ++triId) {
        triRefs.push_back(buildTriRef(mesh, triId));
    }

    std::vector<BuildNode> buildNodes;
    buildNodes.reserve(triCount * 2);

    const int root = buildRecursive(buildNodes, triRefs, 0, triCount);

    FlattenedBvh out;
    out.nodes.reserve(buildNodes.size());
    out.leafTriIndices.reserve(triCount);

    flattenRecursive(root, buildNodes, triRefs, out);
    return out;
}
#pragma once

#define VK_NO_PROTOTYPES
#include <Volk/volk.h>
#include "vk_mem_alloc.h"

#include <cstdint>
#include <vector>

struct AllocatedBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo allocInfo{};
    VkDeviceAddress deviceAddress = 0;
    VkDeviceSize size = 0;
};

struct AccelerationStructure {
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    AllocatedBuffer buffer{};
    uint64_t deviceAddress = 0;
};

struct RtAccelContext {
    VkDevice device = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;

    uint32_t scratchAlignment = 256;
};

struct BlasInputGeometry {
    const std::vector<float>* vertices = nullptr;      // xyzxyz...
    const std::vector<uint32_t>* indices = nullptr;    // 3 per triangle
};

AllocatedBuffer createBuffer(
    const RtAccelContext& ctx,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VmaMemoryUsage memoryUsage,
    VmaAllocationCreateFlags allocFlags = 0
);

void destroyBuffer(const RtAccelContext& ctx, AllocatedBuffer& buf);

void uploadToBuffer(
    const RtAccelContext& ctx,
    const void* data,
    VkDeviceSize size,
    AllocatedBuffer& dst
);

uint64_t getAccelerationStructureDeviceAddress(
    VkDevice device,
    VkAccelerationStructureKHR as
);

AccelerationStructure buildBLAS(
    const RtAccelContext& ctx,
    const BlasInputGeometry& geom,
    AllocatedBuffer& outVertexBuffer,
    AllocatedBuffer& outIndexBuffer
);

AccelerationStructure buildTLAS(
    const RtAccelContext& ctx,
    const AccelerationStructure& blas
);

void destroyAccelerationStructure(
    const RtAccelContext& ctx,
    AccelerationStructure& as
);
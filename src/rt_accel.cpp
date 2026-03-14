#include "rt_accel.h"

#include <cstring>
#include <stdexcept>
#include <array>
#include <iostream>

namespace {
    static VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    uint32_t findMemoryType(
        VkPhysicalDevice,
        uint32_t,
        VkMemoryPropertyFlags
    ) {
        throw std::runtime_error("findMemoryType should not be used with VMA.");
    }

    VkDeviceAddress getBufferDeviceAddress(VkDevice device, VkBuffer buffer) {
        if (buffer == VK_NULL_HANDLE) {
            throw std::runtime_error("getBufferDeviceAddress called with VK_NULL_HANDLE buffer");
        }

        VkBufferDeviceAddressInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        info.buffer = buffer;

        std::cout << "Querying device address for buffer=" << buffer << "\n";

        VkDeviceAddress addr = vkGetBufferDeviceAddressKHR(device, &info);

        std::cout << "Device address result=" << static_cast<unsigned long long>(addr) << "\n";

        if (addr == 0) {
            throw std::runtime_error("vkGetBufferDeviceAddressKHR returned 0");
        }

        return addr;
    }

    VkCommandBuffer beginSingleTimeCommands(const RtAccelContext& ctx) {
        if (ctx.device == VK_NULL_HANDLE) {
            throw std::runtime_error("beginSingleTimeCommands: ctx.device is null");
        }
        if (ctx.commandPool == VK_NULL_HANDLE) {
            throw std::runtime_error("beginSingleTimeCommands: ctx.commandPool is null");
        }
        if (!vkAllocateCommandBuffers) {
            throw std::runtime_error("beginSingleTimeCommands: vkAllocateCommandBuffers is null");
        }

        VkCommandBuffer cmd = VK_NULL_HANDLE;

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.pNext = nullptr;
        allocInfo.commandPool = ctx.commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        std::cout
            << "beginSingleTimeCommands device=" << ctx.device
            << " commandPool=" << ctx.commandPool
            << " vkAllocateCommandBuffers=" << (void*)vkAllocateCommandBuffers
            << "\n";

        VkResult res = vkAllocateCommandBuffers(ctx.device, &allocInfo, &cmd);
        if (res != VK_SUCCESS || cmd == VK_NULL_HANDLE) {
            throw std::runtime_error("vkAllocateCommandBuffers failed");
        }

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.pNext = nullptr;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        beginInfo.pInheritanceInfo = nullptr;

        res = vkBeginCommandBuffer(cmd, &beginInfo);
        if (res != VK_SUCCESS) {
            throw std::runtime_error("vkBeginCommandBuffer failed");
        }

        return cmd;
    }

    void endSingleTimeCommands(const RtAccelContext& ctx, VkCommandBuffer cmd) {
        if (cmd == VK_NULL_HANDLE) {
            throw std::runtime_error("endSingleTimeCommands: cmd is null");
        }

        if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
            throw std::runtime_error("vkEndCommandBuffer failed");
        }

        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.pNext = nullptr;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;

        if (vkQueueSubmit(ctx.graphicsQueue, 1, &submit, VK_NULL_HANDLE) != VK_SUCCESS) {
            throw std::runtime_error("vkQueueSubmit failed");
        }

        if (vkQueueWaitIdle(ctx.graphicsQueue) != VK_SUCCESS) {
            throw std::runtime_error("vkQueueWaitIdle failed");
        }

        vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &cmd);
    }

    void createAccelerationStructureObject(
        const RtAccelContext& ctx,
        VkDeviceSize asBufferSize,
        VkAccelerationStructureTypeKHR type,
        AccelerationStructure& outAs
    ) {
        outAs.buffer = createBuffer(
            ctx,
            asBufferSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
        );

        VkAccelerationStructureCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        createInfo.buffer = outAs.buffer.buffer;
        createInfo.size = asBufferSize;
        createInfo.type = type;

        if (vkCreateAccelerationStructureKHR(ctx.device, &createInfo, nullptr, &outAs.handle) != VK_SUCCESS) {
            throw std::runtime_error("vkCreateAccelerationStructureKHR failed");
        }

        outAs.deviceAddress = getAccelerationStructureDeviceAddress(ctx.device, outAs.handle);
    }

} // namespace

AllocatedBuffer createBuffer(
    const RtAccelContext& ctx,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VmaMemoryUsage memoryUsage,
    VmaAllocationCreateFlags allocFlags)
{
    AllocatedBuffer out{};
    out.size = size;

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = memoryUsage;
    aci.flags = allocFlags;

    const VkResult res = vmaCreateBuffer(
        ctx.allocator,
        &bci,
        &aci,
        &out.buffer,
        &out.allocation,
        &out.allocInfo
    );

    std::cout
        << "createBuffer size=" << static_cast<unsigned long long>(size)
        << " usage=0x" << std::hex << usage
        << " allocFlags=0x" << allocFlags
        << " result=" << std::dec << res
        << " buffer=" << out.buffer
        << " mapped=" << out.allocInfo.pMappedData
        << "\n";

    if (res != VK_SUCCESS || out.buffer == VK_NULL_HANDLE) {
        throw std::runtime_error("vmaCreateBuffer failed");
    }

    if ((usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0) {
        out.deviceAddress = getBufferDeviceAddress(ctx.device, out.buffer);
    }
    else {
        out.deviceAddress = 0;
    }

    return out;
}

void destroyBuffer(const RtAccelContext& ctx, AllocatedBuffer& buf) {
    if (buf.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(ctx.allocator, buf.buffer, buf.allocation);
    }
    buf = {};
}

void uploadToBuffer(
    const RtAccelContext& ctx,
    const void* data,
    VkDeviceSize size,
    AllocatedBuffer& dst
) {
    AllocatedBuffer staging = createBuffer(
        ctx,
        size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT
    );

    std::memcpy(staging.allocInfo.pMappedData, data, static_cast<size_t>(size));

    VkCommandBuffer cmd = beginSingleTimeCommands(ctx);

    VkBufferCopy copy{};
    copy.srcOffset = 0;
    copy.dstOffset = 0;
    copy.size = size;
    vkCmdCopyBuffer(cmd, staging.buffer, dst.buffer, 1, &copy);

    endSingleTimeCommands(ctx, cmd);
    destroyBuffer(ctx, staging);
}

uint64_t getAccelerationStructureDeviceAddress(
    VkDevice device,
    VkAccelerationStructureKHR as
) {
    VkAccelerationStructureDeviceAddressInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    info.accelerationStructure = as;
    return vkGetAccelerationStructureDeviceAddressKHR(device, &info);
}

AccelerationStructure buildBLAS(
    const RtAccelContext& ctx,
    const BlasInputGeometry& geom,
    AllocatedBuffer& outVertexBuffer,
    AllocatedBuffer& outIndexBuffer
) {
    if (!geom.vertices || !geom.indices) {
        throw std::runtime_error("buildBLAS: null geometry pointers");
    }

    const auto& vertices = *geom.vertices;
    const auto& indices = *geom.indices;

    if (vertices.empty() || indices.empty() || (indices.size() % 3) != 0) {
        throw std::runtime_error("buildBLAS: invalid mesh data");
    }

    const VkDeviceSize vertexBytes = sizeof(float) * vertices.size();
    const VkDeviceSize indexBytes = sizeof(uint32_t) * indices.size();
    const uint32_t primitiveCount = static_cast<uint32_t>(indices.size() / 3);

    outVertexBuffer = createBuffer(
        ctx,
        vertexBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
    );

    outIndexBuffer = createBuffer(
        ctx,
        indexBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
    );

    uploadToBuffer(ctx, vertices.data(), vertexBytes, outVertexBuffer);
    uploadToBuffer(ctx, indices.data(), indexBytes, outIndexBuffer);

    VkAccelerationStructureGeometryTrianglesDataKHR trianglesData{};
    trianglesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    trianglesData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    trianglesData.vertexData.deviceAddress = outVertexBuffer.deviceAddress;
    trianglesData.vertexStride = sizeof(float) * 3;
    trianglesData.maxVertex = static_cast<uint32_t>(vertices.size() / 3);
    trianglesData.indexType = VK_INDEX_TYPE_UINT32;
    trianglesData.indexData.deviceAddress = outIndexBuffer.deviceAddress;
    trianglesData.transformData.deviceAddress = 0;

    VkAccelerationStructureGeometryKHR asGeom{};
    asGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    asGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    asGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    asGeom.geometry.triangles = trianglesData;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &asGeom;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    vkGetAccelerationStructureBuildSizesKHR(
        ctx.device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo,
        &primitiveCount,
        &sizeInfo
    );

    VkDeviceSize scratchSize = alignUp(
        sizeInfo.buildScratchSize,
        ctx.scratchAlignment
    );

    AccelerationStructure blas{};
    createAccelerationStructureObject(
        ctx,
        sizeInfo.accelerationStructureSize,
        VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        blas
    );

    AllocatedBuffer scratch = createBuffer(
        ctx,
        scratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
    );

    buildInfo.dstAccelerationStructure = blas.handle;
    buildInfo.scratchData.deviceAddress = scratch.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = primitiveCount;
    rangeInfo.primitiveOffset = 0;
    rangeInfo.firstVertex = 0;
    rangeInfo.transformOffset = 0;

    const VkAccelerationStructureBuildRangeInfoKHR* rangeInfos[] = { &rangeInfo };

    VkCommandBuffer cmd = beginSingleTimeCommands(ctx);

    if (outVertexBuffer.deviceAddress == 0) {
        throw std::runtime_error("BLAS vertex buffer device address is 0");
    }
    if (outIndexBuffer.deviceAddress == 0) {
        throw std::runtime_error("BLAS index buffer device address is 0");
    }
    if (scratch.deviceAddress == 0) {
        throw std::runtime_error("BLAS scratch buffer device address is 0");
    }
    if ((scratch.deviceAddress % ctx.scratchAlignment) != 0) {
        throw std::runtime_error("BLAS scratch address is misaligned");
    }

    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, rangeInfos);
    endSingleTimeCommands(ctx, cmd);

    destroyBuffer(ctx, scratch);
    return blas;
}

AccelerationStructure buildTLAS(
    const RtAccelContext& ctx,
    const AccelerationStructure& blas
) {
    VkTransformMatrixKHR transformMatrix = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };

    VkAccelerationStructureInstanceKHR instance{};
    instance.transform = transformMatrix;
    instance.instanceCustomIndex = 0;
    instance.mask = 0xFF;
    instance.instanceShaderBindingTableRecordOffset = 0;
    instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instance.accelerationStructureReference = blas.deviceAddress;

    AllocatedBuffer instanceBuffer = createBuffer(
        ctx,
        sizeof(VkAccelerationStructureInstanceKHR),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
    );

    uploadToBuffer(ctx, &instance, sizeof(instance), instanceBuffer);

    VkAccelerationStructureGeometryInstancesDataKHR instancesData{};
    instancesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instancesData.arrayOfPointers = VK_FALSE;
    instancesData.data.deviceAddress = instanceBuffer.deviceAddress;

    VkAccelerationStructureGeometryKHR asGeom{};
    asGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    asGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    asGeom.geometry.instances = instancesData;

    uint32_t primitiveCount = 1;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &asGeom;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(
        ctx.device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo,
        &primitiveCount,
        &sizeInfo
    );

    VkDeviceSize scratchSize = alignUp(
        sizeInfo.buildScratchSize,
        ctx.scratchAlignment
    );

    AccelerationStructure tlas{};
    createAccelerationStructureObject(
        ctx,
        sizeInfo.accelerationStructureSize,
        VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        tlas
    );

    AllocatedBuffer scratch = createBuffer(
        ctx,
        scratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
    );

    buildInfo.dstAccelerationStructure = tlas.handle;
    buildInfo.scratchData.deviceAddress = scratch.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = 1;
    rangeInfo.primitiveOffset = 0;
    rangeInfo.firstVertex = 0;
    rangeInfo.transformOffset = 0;

    const VkAccelerationStructureBuildRangeInfoKHR* rangeInfos[] = { &rangeInfo };

    VkCommandBuffer cmd = beginSingleTimeCommands(ctx);

    if (instanceBuffer.deviceAddress == 0) {
        throw std::runtime_error("TLAS instance buffer device address is 0");
    }
    if (scratch.deviceAddress == 0) {
        throw std::runtime_error("TLAS scratch buffer device address is 0");
    }
    if ((scratch.deviceAddress % ctx.scratchAlignment) != 0) {
        throw std::runtime_error("TLAS scratch address is misaligned");
    }

    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, rangeInfos);
    endSingleTimeCommands(ctx, cmd);

    destroyBuffer(ctx, scratch);
    destroyBuffer(ctx, instanceBuffer);
    return tlas;
}

void destroyAccelerationStructure(
    const RtAccelContext& ctx,
    AccelerationStructure& as
) {
    if (as.handle != VK_NULL_HANDLE) {
        vkDestroyAccelerationStructureKHR(ctx.device, as.handle, nullptr);
    }
    destroyBuffer(ctx, as.buffer);
    as = {};
}
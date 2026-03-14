#define VK_NO_PROTOTYPES
#include <Volk/volk.h>
#include <GLFW/glfw3.h>
#include "obj_loader.h"
#include "rt_accel.h"
#include "bvh.h"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <cmath>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <vector>

struct RenderSettings {
    bool sunEnabled = true;

    float sunDirection[3] = { 0.4f, -1.0f, 0.3f };
    float sunIntensity = 1.0f;
    float sunColor[3] = { 1.0f, 0.95f, 0.9f };

    int maxBounces = 4;

    bool fogEnabled = false;
    float fogDensity = 0.0f;
    float fogG = 0.0f; // Henyey-Greenstein anisotropy

    bool enableNEE = true;
    bool accumulate = true;
};

struct CameraUBO {
    float camPos[4];
    float camForward[4];
    float camRight[4];
    float camUp[4];
    float camData[4];
};

struct CameraState {
    float px = 0.0f;
    float py = 0.0f;
    float pz = -2.0f;

    float yaw = 0.0f;
    float pitch = 0.0f;

    float moveSpeed = 0.0003f;
    float rotSpeed = 0.001f;
    float fovY = 45.0f * 3.1415926535f / 180.0f;
};

struct GpuPathState {
    float ro[4];
    float rd[4];
    float throughput[4];
    float radiance[4];
    uint32_t pixelIndex;
    uint32_t alive;
    uint32_t bounce;
    uint32_t pad0;
};

struct FrameParams {
    uint32_t width;
    uint32_t height;
    uint32_t pathCount;
    uint32_t frameIndex;

    uint32_t emissiveTriangleCount;
    uint32_t sunEnabled;
    uint32_t fogEnabled;
    uint32_t maxBounces;

    float camPos[4];
    float camForward[4];
    float camRight[4];
    float camUp[4];
    float camData[4];

    float sunDirIntensity[4];   // xyz = dir, w = intensity
    float sunColor[4];          // xyz = color

    float fogParams[4];         // x = density, y = g, z/w unused
};

static void uploadMaterials(
    RtAccelContext& rtCtx,
    const MeshData& mesh,
    AllocatedBuffer& materialBuffer
) {
    uploadToBuffer(
        rtCtx,
        mesh.materials.data(),
        sizeof(Material) * mesh.materials.size(),
        materialBuffer
    );
}

static bool drawMaterialEditor(MeshData& mesh, int& selectedMaterial) {
    bool changed = false;

    ImGui::Begin("Materials");

    if (mesh.materials.empty()) {
        ImGui::Text("No materials.");
        ImGui::End();
        return false;
    }

    if (selectedMaterial < 0) selectedMaterial = 0;
    if (selectedMaterial >= static_cast<int>(mesh.materials.size())) {
        selectedMaterial = static_cast<int>(mesh.materials.size()) - 1;
    }

    std::vector<const char*> names;
    names.reserve(mesh.materials.size());

    static std::vector<std::string> labels;
    labels.clear();
    labels.reserve(mesh.materials.size());

    for (size_t i = 0; i < mesh.materials.size(); ++i) {
        labels.push_back("Material " + std::to_string(i));
        names.push_back(labels.back().c_str());
    }

    ImGui::Text("Material count: %d", static_cast<int>(mesh.materials.size()));
    ImGui::Separator();

    ImGui::Combo("Selected", &selectedMaterial, names.data(), static_cast<int>(names.size()));

    Material& m = mesh.materials[selectedMaterial];

    changed |= ImGui::ColorEdit3("Albedo", m.albedo_smoothness);
    changed |= ImGui::SliderFloat("Smoothness", &m.albedo_smoothness[3], 0.0f, 1.0f);

    changed |= ImGui::ColorEdit3("Emissive Color", m.emissive_emissiveIntensity);
    changed |= ImGui::SliderFloat("Emissive Intensity", &m.emissive_emissiveIntensity[3], 0.0f, 50.0f);

    changed |= ImGui::SliderFloat("Metallic", &m.metallic_smoothShading_transmission_ior[0], 0.0f, 1.0f);
    changed |= ImGui::SliderFloat("Smooth Shading", &m.metallic_smoothShading_transmission_ior[1], 0.0f, 1.0f);
    changed |= ImGui::SliderFloat("Transmission", &m.metallic_smoothShading_transmission_ior[2], 0.0f, 1.0f);
    changed |= ImGui::SliderFloat("IOR", &m.metallic_smoothShading_transmission_ior[3], 1.0f, 3.0f);

    changed |= ImGui::SliderFloat("Dispersion", &m.dispersion_thinFilmThickness_thinFilmIor_thinFilmSubstrateEta[0], 0.0f, 2.0f);
    changed |= ImGui::SliderFloat("Thin Film Thickness", &m.dispersion_thinFilmThickness_thinFilmIor_thinFilmSubstrateEta[1], 0.0f, 2000.0f);
    changed |= ImGui::SliderFloat("Thin Film IOR", &m.dispersion_thinFilmThickness_thinFilmIor_thinFilmSubstrateEta[2], 1.0f, 3.0f);
    changed |= ImGui::SliderFloat("Thin Film Substrate Eta", &m.dispersion_thinFilmThickness_thinFilmIor_thinFilmSubstrateEta[3], 1.0f, 5.0f);

    changed |= ImGui::SliderFloat("Thin Film Substrate K", &m.thinFilmSubstrateK_pad[0], 0.0f, 10.0f);

    ImGui::End();
    return changed;
}

static bool drawRendererSettingsEditor(RenderSettings& s) {
    bool changed = false;

    ImGui::Begin("Renderer");

    changed |= ImGui::Checkbox("Accumulate", &s.accumulate);
    changed |= ImGui::Checkbox("Enable NEE", &s.enableNEE);

    changed |= ImGui::SliderInt("Max Bounces", &s.maxBounces, 1, 32);

    ImGui::Separator();
    ImGui::Text("Sun");

    changed |= ImGui::Checkbox("Sun Enabled", &s.sunEnabled);
    changed |= ImGui::SliderFloat3("Sun Direction", s.sunDirection, -1.0f, 1.0f);
    changed |= ImGui::ColorEdit3("Sun Color", s.sunColor);
    changed |= ImGui::SliderFloat("Sun Intensity", &s.sunIntensity, 0.0f, 50.0f);

    ImGui::Separator();
    ImGui::Text("Fog");

    changed |= ImGui::Checkbox("Fog Enabled", &s.fogEnabled);
    changed |= ImGui::SliderFloat("Fog Density", &s.fogDensity, 0.0f, 10.0f, "%.4f", ImGuiSliderFlags_Logarithmic);
    changed |= ImGui::SliderFloat("Fog G", &s.fogG, -0.99f, 0.99f);

    ImGui::End();
    return changed;
}

static void check_vk_result(VkResult err)
{
    if (err == VK_SUCCESS)
        return;

    std::cerr << "Vulkan error: " << err << "\n";

    if (err < 0)
        throw std::runtime_error("ImGui Vulkan call failed");
}

static void normalize3(float& x, float& y, float& z) {
    const float len = std::sqrt(x * x + y * y + z * z);
    if (len > 0.0f) {
        x /= len;
        y /= len;
        z /= len;
    }
}

static void cross3(
    float ax, float ay, float az,
    float bx, float by, float bz,
    float& rx, float& ry, float& rz
) {
    rx = ay * bz - az * by;
    ry = az * bx - ax * bz;
    rz = ax * by - ay * bx;
}

static void buildCameraBasis(
    const CameraState& cam,
    float forward[3],
    float right[3],
    float up[3]
) {
    const float cp = std::cos(cam.pitch);
    forward[0] = std::sin(cam.yaw) * cp;
    forward[1] = std::sin(cam.pitch);
    forward[2] = std::cos(cam.yaw) * cp;
    normalize3(forward[0], forward[1], forward[2]);

    const float worldUp[3] = { 0.0f, 1.0f, 0.0f };

    cross3(
        forward[0], forward[1], forward[2],
        worldUp[0], worldUp[1], worldUp[2],
        right[0], right[1], right[2]
    );
    normalize3(right[0], right[1], right[2]);

    cross3(
        right[0], right[1], right[2],
        forward[0], forward[1], forward[2],
        up[0], up[1], up[2]
    );
    normalize3(up[0], up[1], up[2]);
}

static std::vector<float> buildVertexNormals(const MeshData& mesh) {
    std::vector<float> normals(mesh.vertices.size() * 3, 0.0f);

    const uint32_t triCount = static_cast<uint32_t>(mesh.indices.size() / 3);
    for (uint32_t t = 0; t < triCount; ++t) {
        uint32_t i0 = mesh.indices[t * 3 + 0];
        uint32_t i1 = mesh.indices[t * 3 + 1];
        uint32_t i2 = mesh.indices[t * 3 + 2];

        const Vertex& a = mesh.vertices[i0];
        const Vertex& b = mesh.vertices[i1];
        const Vertex& c = mesh.vertices[i2];

        float e1x = b.x - a.x;
        float e1y = b.y - a.y;
        float e1z = b.z - a.z;

        float e2x = c.x - a.x;
        float e2y = c.y - a.y;
        float e2z = c.z - a.z;

        float nx = e1y * e2z - e1z * e2y;
        float ny = e1z * e2x - e1x * e2z;
        float nz = e1x * e2y - e1y * e2x;

        normals[i0 * 3 + 0] += nx;
        normals[i0 * 3 + 1] += ny;
        normals[i0 * 3 + 2] += nz;

        normals[i1 * 3 + 0] += nx;
        normals[i1 * 3 + 1] += ny;
        normals[i1 * 3 + 2] += nz;

        normals[i2 * 3 + 0] += nx;
        normals[i2 * 3 + 1] += ny;
        normals[i2 * 3 + 2] += nz;
    }

    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
        float& x = normals[i * 3 + 0];
        float& y = normals[i * 3 + 1];
        float& z = normals[i * 3 + 2];
        float len = std::sqrt(x * x + y * y + z * z);
        if (len > 1e-20f) {
            x /= len;
            y /= len;
            z /= len;
        }
        else {
            x = 0.0f;
            y = 1.0f;
            z = 0.0f;
        }
    }

    return normals;
}

static bool updateCameraFromInput(GLFWwindow* window, CameraState& cam) {
    bool changed = false;

    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) { cam.yaw -= cam.rotSpeed; changed = true; }
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) { cam.yaw += cam.rotSpeed; changed = true; }
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) { cam.pitch += cam.rotSpeed; changed = true; }
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) { cam.pitch -= cam.rotSpeed; changed = true; }

    const float limit = 1.55f;
    if (cam.pitch > limit) cam.pitch = limit;
    if (cam.pitch < -limit) cam.pitch = -limit;

    float forward[3], right[3], up[3];
    buildCameraBasis(cam, forward, right, up);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        cam.px += forward[0] * cam.moveSpeed;
        cam.py += forward[1] * cam.moveSpeed;
        cam.pz += forward[2] * cam.moveSpeed;
        changed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        cam.px -= forward[0] * cam.moveSpeed;
        cam.py -= forward[1] * cam.moveSpeed;
        cam.pz -= forward[2] * cam.moveSpeed;
        changed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        cam.px += right[0] * cam.moveSpeed;
        cam.py += right[1] * cam.moveSpeed;
        cam.pz += right[2] * cam.moveSpeed;
        changed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        cam.px -= right[0] * cam.moveSpeed;
        cam.py -= right[1] * cam.moveSpeed;
        cam.pz -= right[2] * cam.moveSpeed;
        changed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        cam.py += cam.moveSpeed;
        changed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
        cam.py -= cam.moveSpeed;
        changed = true;
    }

    return changed;
}

static void appendCornellBox(
    MeshData& mesh,
    float minX, float minY, float minZ,
    float maxX, float maxY, float maxZ,
    uint32_t startMaterialIndex
) {
    auto addVertex = [&](float x, float y, float z) -> uint32_t {
        uint32_t idx = static_cast<uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back(Vertex{ x, y, z });
        return idx;
        };

    auto addTri = [&](uint32_t a, uint32_t b, uint32_t c, uint32_t materialIndex) {
        mesh.indices.push_back(a);
        mesh.indices.push_back(b);
        mesh.indices.push_back(c);
        mesh.triangleMaterialIndices.push_back(materialIndex);
        };

    auto addQuad = [&](uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t materialIndex) {
        addTri(a, b, c, materialIndex);
        addTri(a, c, d, materialIndex);
        };

    uint32_t v0 = addVertex(minX, minY, minZ);
    uint32_t v1 = addVertex(maxX, minY, minZ);
    uint32_t v2 = addVertex(maxX, maxY, minZ);
    uint32_t v3 = addVertex(minX, maxY, minZ);

    uint32_t v4 = addVertex(minX, minY, maxZ);
    uint32_t v5 = addVertex(maxX, minY, maxZ);
    uint32_t v6 = addVertex(maxX, maxY, maxZ);
    uint32_t v7 = addVertex(minX, maxY, maxZ);

    uint32_t mat = startMaterialIndex;

    // floor
    addQuad(v0, v4, v5, v1, mat++);
    // ceiling
    addQuad(v3, v2, v6, v7, mat++);
    // back wall
    addQuad(v7, v6, v5, v4, mat++);
    // left wall
    addQuad(v3, v7, v4, v0, mat++);
    // right wall
    addQuad(v2, v1, v5, v6, mat++);
}

static CameraUBO makeCameraUBO(const CameraState& cam, uint32_t width, uint32_t height) {
    float forward[3], right[3], up[3];
    buildCameraBasis(cam, forward, right, up);

    CameraUBO ubo{};
    ubo.camPos[0] = cam.px;
    ubo.camPos[1] = cam.py;
    ubo.camPos[2] = cam.pz;
    ubo.camPos[3] = 0.0f;

    ubo.camForward[0] = forward[0];
    ubo.camForward[1] = forward[1];
    ubo.camForward[2] = forward[2];
    ubo.camForward[3] = 0.0f;

    ubo.camRight[0] = right[0];
    ubo.camRight[1] = right[1];
    ubo.camRight[2] = right[2];
    ubo.camRight[3] = 0.0f;

    ubo.camUp[0] = up[0];
    ubo.camUp[1] = up[1];
    ubo.camUp[2] = up[2];
    ubo.camUp[3] = 0.0f;

    ubo.camData[0] = std::tan(0.5f * cam.fovY);
    ubo.camData[1] = static_cast<float>(width) / static_cast<float>(height);
    ubo.camData[2] = 0.0f;
    ubo.camData[3] = 0.0f;

    return ubo;
}

static MeshData loadObjMesh(const std::string& path) {
    const std::vector<std::string> lines = loadLines(path);

    ObjMesh obj = parseOBJ(
        lines,
        1.0f,    // MODEL_SCALE
        0.0f,    // MODEL_TRANSLATE[0]
        -0.025f, // MODEL_TRANSLATE[1]
        0.0f     // MODEL_TRANSLATE[2]
    );

    rotateModelAroundCenter(
        obj.vertices,
        0.0f,               // pitch
        -3.14159265f * 0.5f, // yaw
        0.0f                // roll
    );

    MeshData mesh;
    mesh.vertices.reserve(obj.vertices.size() / 3);
    mesh.indices = obj.triangles;

    for (size_t i = 0; i + 2 < obj.vertices.size(); i += 3) {
        mesh.vertices.push_back(Vertex{
            obj.vertices[i + 0],
            obj.vertices[i + 1],
            obj.vertices[i + 2]
            });
    }

    if (mesh.indices.size() % 3 != 0) {
        throw std::runtime_error("OBJ index count is not a multiple of 3.");
    }

    Material defaultMat{};
    defaultMat.albedo_smoothness[0] = 0.1f;
    defaultMat.albedo_smoothness[1] = 0.8f;
    defaultMat.albedo_smoothness[2] = 0.8f;
    defaultMat.albedo_smoothness[3] = 0.0f;

    defaultMat.emissive_emissiveIntensity[0] = 0.0f;
    defaultMat.emissive_emissiveIntensity[1] = 0.0f;
    defaultMat.emissive_emissiveIntensity[2] = 0.0f;
    defaultMat.emissive_emissiveIntensity[3] = 0.0f;

    defaultMat.metallic_smoothShading_transmission_ior[0] = 0.0f;
    defaultMat.metallic_smoothShading_transmission_ior[1] = 1.0f;
    defaultMat.metallic_smoothShading_transmission_ior[2] = 0.0f;
    defaultMat.metallic_smoothShading_transmission_ior[3] = 1.5f;

    defaultMat.dispersion_thinFilmThickness_thinFilmIor_thinFilmSubstrateEta[0] = 0.0f;
    defaultMat.dispersion_thinFilmThickness_thinFilmIor_thinFilmSubstrateEta[1] = 0.0f;
    defaultMat.dispersion_thinFilmThickness_thinFilmIor_thinFilmSubstrateEta[2] = 1.0f;
    defaultMat.dispersion_thinFilmThickness_thinFilmIor_thinFilmSubstrateEta[3] = 1.0f;
    defaultMat.thinFilmSubstrateK_pad[0] = 0.0f;

    mesh.materials.push_back(defaultMat);

    const uint32_t triCount = static_cast<uint32_t>(mesh.indices.size() / 3);
    mesh.triangleMaterialIndices.resize(triCount, 0u);

    return mesh;
}

static std::vector<uint32_t> readSpvFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open SPIR-V file: " + path);
    }

    const std::streamsize size = file.tellg();
    if (size <= 0 || (size % 4) != 0) {
        throw std::runtime_error("Invalid SPIR-V file size: " + path);
    }
    file.seekg(0, std::ios::beg);

    std::vector<uint32_t> data(static_cast<size_t>(size / 4));
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("Failed to read SPIR-V file: " + path);
    }
    return data;
}

struct QueueFamilyIndices {
    bool hasGraphics = false;
    bool hasPresent = false;
    uint32_t graphics = 0;
    uint32_t present = 0;

    bool complete() const { return hasGraphics && hasPresent; }
};

static QueueFamilyIndices findQueueFamilies(VkPhysicalDevice gpu, VkSurfaceKHR surface) {
    QueueFamilyIndices out{};

    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, props.data());

    std::cout << "Queue family count: " << count << "\n";

    for (uint32_t i = 0; i < count; ++i) {
        VkBool32 supported = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(gpu, i, surface, &supported);

        std::cout
            << "Queue " << i
            << " flags=" << props[i].queueFlags
            << " graphics=" << ((props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) ? "yes" : "no")
            << " present=" << (supported ? "yes" : "no")
            << "\n";

        if ((props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && !out.hasGraphics) {
            out.hasGraphics = true;
            out.graphics = i;
        }

        if (supported && !out.hasPresent) {
            out.hasPresent = true;
            out.present = i;
        }
    }

    return out;
}

struct SwapchainSupport {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

static SwapchainSupport querySwapchainSupport(VkPhysicalDevice gpu, VkSurfaceKHR surface) {
    SwapchainSupport out;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu, surface, &out.capabilities);

    uint32_t count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &count, nullptr);
    out.formats.resize(count);
    if (count) vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &count, out.formats.data());

    count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface, &count, nullptr);
    out.presentModes.resize(count);
    if (count) vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface, &count, out.presentModes.data());

    return out;
}

static VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) {
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return f;
        }
    }
    return formats.at(0);
}

static VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& modes) {
    for (auto m : modes) {
        if (m == VK_PRESENT_MODE_MAILBOX_KHR) return m;
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

static VkExtent2D chooseExtent(const VkSurfaceCapabilitiesKHR& caps, GLFWwindow* window) {
    if (caps.currentExtent.width != UINT32_MAX) return caps.currentExtent;

    int w = 0, h = 0;
    glfwGetFramebufferSize(window, &w, &h);

    VkExtent2D extent{};
    extent.width = static_cast<uint32_t>(w);
    extent.height = static_cast<uint32_t>(h);
    extent.width = std::max(caps.minImageExtent.width, std::min(caps.maxImageExtent.width, extent.width));
    extent.height = std::max(caps.minImageExtent.height, std::min(caps.maxImageExtent.height, extent.height));
    return extent;
}

static VkShaderModule createShaderModule(VkDevice device, const std::vector<uint32_t>& code) {
    VkShaderModuleCreateInfo ci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    ci.codeSize = code.size() * sizeof(uint32_t);
    ci.pCode = code.data();

    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &ci, nullptr, &module) != VK_SUCCESS) {
        throw std::runtime_error("vkCreateShaderModule failed");
    }
    return module;
}

static float luminance(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

static std::vector<EmissiveTriangle> buildEmissiveTriangles(const MeshData& mesh) {
    std::vector<EmissiveTriangle> out;
    const uint32_t triCount = static_cast<uint32_t>(mesh.indices.size() / 3);

    float totalWeight = 0.0f;

    for (uint32_t tri = 0; tri < triCount; ++tri) {
        uint32_t matIndex = mesh.triangleMaterialIndices[tri];
        const Material& mat = mesh.materials[matIndex];

        float er = mat.emissive_emissiveIntensity[0] * mat.emissive_emissiveIntensity[3];
        float eg = mat.emissive_emissiveIntensity[1] * mat.emissive_emissiveIntensity[3];
        float eb = mat.emissive_emissiveIntensity[2] * mat.emissive_emissiveIntensity[3];

        float lum = luminance(er, eg, eb);
        if (lum <= 0.0f) continue;

        uint32_t i0 = mesh.indices[tri * 3 + 0];
        uint32_t i1 = mesh.indices[tri * 3 + 1];
        uint32_t i2 = mesh.indices[tri * 3 + 2];

        const Vertex& a = mesh.vertices[i0];
        const Vertex& b = mesh.vertices[i1];
        const Vertex& c = mesh.vertices[i2];

        float abx = b.x - a.x, aby = b.y - a.y, abz = b.z - a.z;
        float acx = c.x - a.x, acy = c.y - a.y, acz = c.z - a.z;

        float cx = aby * acz - abz * acy;
        float cy = abz * acx - abx * acz;
        float cz = abx * acy - aby * acx;

        float area = 0.5f * std::sqrt(cx * cx + cy * cy + cz * cz);
        if (area <= 0.0f) continue;

        EmissiveTriangle e{};
        e.triIndex = tri;
        e.area = area;
        e.weight = area * lum;
        e.cdf = 0.0f;

        totalWeight += e.weight;
        out.push_back(e);
    }

    if (totalWeight > 0.0f) {
        float accum = 0.0f;
        for (auto& e : out) {
            accum += e.weight / totalWeight;
            e.cdf = accum;
        }
        out.back().cdf = 1.0f;
    }

    return out;
}

int main() {
    try {
        const std::string objPath = "Shared/stanfordbunny.obj";
        MeshData mesh = loadObjMesh(objPath);
        std::cout << "Loaded OBJ: " << objPath << "\n";
        std::cout << "Vertices: " << mesh.vertices.size() << "\n";
        std::cout << "Triangles: " << (mesh.indices.size() / 3) << "\n";

        size_t badIndices = 0;
        for (size_t i = 0; i < mesh.indices.size(); ++i) {
            if (mesh.indices[i] >= mesh.vertices.size()) {
                ++badIndices;
            }
        }
        std::cout << "Bad indices: " << badIndices << "\n";

        size_t degenerate = 0;
        for (size_t t = 0; t < mesh.indices.size(); t += 3) {
            const auto& a = mesh.vertices[mesh.indices[t + 0]];
            const auto& b = mesh.vertices[mesh.indices[t + 1]];
            const auto& c = mesh.vertices[mesh.indices[t + 2]];

            float abx = b.x - a.x, aby = b.y - a.y, abz = b.z - a.z;
            float acx = c.x - a.x, acy = c.y - a.y, acz = c.z - a.z;

            float nx = aby * acz - abz * acy;
            float ny = abz * acx - abx * acz;
            float nz = abx * acy - aby * acx;

            float area2 = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (area2 < 1e-10f) ++degenerate;
        }
        std::cout << "Degenerate triangles: " << degenerate << "\n";

        uint32_t boxMaterialStart = static_cast<uint32_t>(mesh.materials.size());

        Material floorMat{};
        floorMat.albedo_smoothness[0] = 0.75f;
        floorMat.albedo_smoothness[1] = 0.75f;
        floorMat.albedo_smoothness[2] = 0.75f;
        floorMat.albedo_smoothness[3] = 0.0f;
        floorMat.metallic_smoothShading_transmission_ior[1] = 0.0f;
        floorMat.metallic_smoothShading_transmission_ior[3] = 1.5f;

        Material ceilMat = floorMat;

        Material backMat = floorMat;

        Material leftMat = floorMat;
        leftMat.albedo_smoothness[0] = 0.75f;
        leftMat.albedo_smoothness[1] = 0.15f;
        leftMat.albedo_smoothness[2] = 0.15f;

        Material rightMat = floorMat;
        rightMat.emissive_emissiveIntensity[0] = 1.0f;
        rightMat.emissive_emissiveIntensity[1] = 0.95f;
        rightMat.emissive_emissiveIntensity[2] = 0.9f;
        rightMat.emissive_emissiveIntensity[3] = 1.0f;

        mesh.materials.push_back(floorMat);
        mesh.materials.push_back(ceilMat);
        mesh.materials.push_back(backMat);
        mesh.materials.push_back(leftMat);
        mesh.materials.push_back(rightMat);

        appendCornellBox(
            mesh,
            -0.3f, 0.0f, -0.3f,
            0.3f, 0.2f, 0.3f,
            boxMaterialStart
        );

        std::vector<EmissiveTriangle> emissiveTriangles = buildEmissiveTriangles(mesh);
        std::cout << "Emissive triangles: " << emissiveTriangles.size() << "\n";

        std::cout << "Material count: " << mesh.materials.size() << "\n";
        std::cout << "Triangle material count: " << mesh.triangleMaterialIndices.size() << "\n";
        std::cout << "Triangle count: " << (mesh.indices.size() / 3) << "\n";

        FlattenedBvh bvh = buildBvh(mesh);

        std::cout << "BVH nodes: " << bvh.nodes.size() << "\n";
        std::cout << "BVH leaf tri refs: " << bvh.leafTriIndices.size() << "\n";

        if (!glfwInit()) {
            throw std::runtime_error("glfwInit failed");
        }
        if (!glfwVulkanSupported()) {
            throw std::runtime_error("GLFW Vulkan not supported");
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        GLFWwindow* window = glfwCreateWindow(1280, 720, "Vulkan Pathtracer Scaffold", nullptr, nullptr);
        if (!window) {
            throw std::runtime_error("glfwCreateWindow failed");
        }

        if (volkInitialize() != VK_SUCCESS) {
            throw std::runtime_error("volkInitialize failed");
        }

        uint32_t extCount = 0;
        const char** glfwExts = glfwGetRequiredInstanceExtensions(&extCount);
        if (!glfwExts || extCount == 0) {
            throw std::runtime_error("glfwGetRequiredInstanceExtensions failed");
        }

        std::vector<const char*> instanceExtensions(glfwExts, glfwExts + extCount);

        VkApplicationInfo appInfo{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
        appInfo.pApplicationName = "Vulkan Pathtracer Scaffold";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "None";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo ici{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        ici.pApplicationInfo = &appInfo;
        ici.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
        ici.ppEnabledExtensionNames = instanceExtensions.data();

        VkInstance instance = VK_NULL_HANDLE;
        if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("vkCreateInstance failed");
        }
        volkLoadInstance(instance);

        VkSurfaceKHR surface = VK_NULL_HANDLE;
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("glfwCreateWindowSurface failed");
        }

        uint32_t gpuCount = 0;
        vkEnumeratePhysicalDevices(instance, &gpuCount, nullptr);
        if (gpuCount == 0) {
            throw std::runtime_error("No Vulkan physical devices found");
        }
        std::vector<VkPhysicalDevice> gpus(gpuCount);
        vkEnumeratePhysicalDevices(instance, &gpuCount, gpus.data());

        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        QueueFamilyIndices queues{};

        for (VkPhysicalDevice gpu : gpus) {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(gpu, &props);

            std::cout << "Checking GPU: " << props.deviceName << "\n";

            QueueFamilyIndices q = findQueueFamilies(gpu, surface);

            if (q.complete()) {
                physicalDevice = gpu;
                queues = q;
                std::cout << "Selected GPU: " << props.deviceName << "\n";
                break;
            }
        }
        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("No suitable GPU with graphics+present support found");
        }

        VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps{};
        asProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;

        VkPhysicalDeviceProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &asProps;

        vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

        std::cout
            << "minAccelerationStructureScratchOffsetAlignment = "
            << asProps.minAccelerationStructureScratchOffsetAlignment
            << "\n";

        VkPhysicalDeviceProperties gpuProps{};
        vkGetPhysicalDeviceProperties(physicalDevice, &gpuProps);
        std::cout << "Using GPU: " << gpuProps.deviceName << "\n";

        std::vector<VkDeviceQueueCreateInfo> queueCIs;
        std::vector<uint32_t> uniqueQueues = { queues.graphics, queues.present };
        std::sort(uniqueQueues.begin(), uniqueQueues.end());
        uniqueQueues.erase(std::unique(uniqueQueues.begin(), uniqueQueues.end()), uniqueQueues.end());
        const float queuePriority = 1.0f;
        for (uint32_t qf : uniqueQueues) {
            VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
            qci.queueFamilyIndex = qf;
            qci.queueCount = 1;
            qci.pQueuePriorities = &queuePriority;
            queueCIs.push_back(qci);
        }

        const std::vector<const char*> deviceExtensions = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
            VK_KHR_RAY_QUERY_EXTENSION_NAME,
            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
            VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
            VK_KHR_SPIRV_1_4_EXTENSION_NAME,
            VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
        };

        VkPhysicalDeviceBufferDeviceAddressFeatures bdaFeatures{};
        bdaFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
        bdaFeatures.bufferDeviceAddress = VK_TRUE;

        VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{};
        asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
        asFeatures.accelerationStructure = VK_TRUE;
        asFeatures.pNext = &bdaFeatures;

        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeatures{};
        rtPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
        rtPipelineFeatures.rayTracingPipeline = VK_TRUE;
        rtPipelineFeatures.pNext = &asFeatures;

        VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{};
        rayQueryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
        rayQueryFeatures.rayQuery = VK_TRUE;
        rayQueryFeatures.pNext = &rtPipelineFeatures;

        VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        dci.pNext = &rayQueryFeatures;
        dci.queueCreateInfoCount = static_cast<uint32_t>(queueCIs.size());
        dci.pQueueCreateInfos = queueCIs.data();
        dci.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        dci.ppEnabledExtensionNames = deviceExtensions.data();

        VkPhysicalDeviceBufferDeviceAddressFeatures bdaCheck{};
        bdaCheck.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;

        VkPhysicalDeviceAccelerationStructureFeaturesKHR asCheck{};
        asCheck.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
        asCheck.pNext = &bdaCheck;

        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtCheck{};
        rtCheck.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
        rtCheck.pNext = &asCheck;

        VkPhysicalDeviceFeatures2 feats2{};
        feats2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        feats2.pNext = &rtCheck;

        vkGetPhysicalDeviceFeatures2(physicalDevice, &feats2);

        std::cout
            << "bufferDeviceAddress: " << bdaCheck.bufferDeviceAddress << "\n"
            << "accelerationStructure: " << asCheck.accelerationStructure << "\n"
            << "rayTracingPipeline: " << rtCheck.rayTracingPipeline << "\n";

        VkDevice device = VK_NULL_HANDLE;
        if (vkCreateDevice(physicalDevice, &dci, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("vkCreateDevice failed");
        }
        volkLoadDevice(device);
        std::cout << "vkGetBufferDeviceAddressKHR: " << (void*)vkGetBufferDeviceAddressKHR << "\n";
        std::cout << "vkCreateAccelerationStructureKHR: " << (void*)vkCreateAccelerationStructureKHR << "\n";
        std::cout << "vkCmdBuildAccelerationStructuresKHR: " << (void*)vkCmdBuildAccelerationStructuresKHR << "\n";

        if (!vkGetBufferDeviceAddressKHR) {
            throw std::runtime_error("vkGetBufferDeviceAddressKHR was not loaded.");
        }
        if (!vkCreateAccelerationStructureKHR || !vkCmdBuildAccelerationStructuresKHR) {
            throw std::runtime_error("Ray tracing acceleration structure functions were not loaded.");
        }

        VkDescriptorPool imguiPool = VK_NULL_HANDLE;
        {
            std::array<VkDescriptorPoolSize, 11> poolSizes = { {
                { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
                { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
                { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
                { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
                { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
            } };

            VkDescriptorPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
            poolInfo.maxSets = 1000 * static_cast<uint32_t>(poolSizes.size());
            poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
            poolInfo.pPoolSizes = poolSizes.data();

            if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &imguiPool) != VK_SUCCESS) {
                throw std::runtime_error("vkCreateDescriptorPool(imguiPool) failed");
            }
        }

        VmaAllocator allocator = VK_NULL_HANDLE;

        VmaVulkanFunctions vmaFuncs{};
        vmaFuncs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
        vmaFuncs.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
        vmaFuncs.vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties;
        vmaFuncs.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
        vmaFuncs.vkAllocateMemory = vkAllocateMemory;
        vmaFuncs.vkFreeMemory = vkFreeMemory;
        vmaFuncs.vkMapMemory = vkMapMemory;
        vmaFuncs.vkUnmapMemory = vkUnmapMemory;
        vmaFuncs.vkFlushMappedMemoryRanges = vkFlushMappedMemoryRanges;
        vmaFuncs.vkInvalidateMappedMemoryRanges = vkInvalidateMappedMemoryRanges;
        vmaFuncs.vkBindBufferMemory = vkBindBufferMemory;
        vmaFuncs.vkBindImageMemory = vkBindImageMemory;
        vmaFuncs.vkGetBufferMemoryRequirements = vkGetBufferMemoryRequirements;
        vmaFuncs.vkGetImageMemoryRequirements = vkGetImageMemoryRequirements;
        vmaFuncs.vkCreateBuffer = vkCreateBuffer;
        vmaFuncs.vkDestroyBuffer = vkDestroyBuffer;
        vmaFuncs.vkCreateImage = vkCreateImage;
        vmaFuncs.vkDestroyImage = vkDestroyImage;
        vmaFuncs.vkCmdCopyBuffer = vkCmdCopyBuffer;
        vmaFuncs.vkGetBufferMemoryRequirements2KHR = vkGetBufferMemoryRequirements2;
        vmaFuncs.vkGetImageMemoryRequirements2KHR = vkGetImageMemoryRequirements2;
        vmaFuncs.vkBindBufferMemory2KHR = vkBindBufferMemory2;
        vmaFuncs.vkBindImageMemory2KHR = vkBindImageMemory2;
        vmaFuncs.vkGetPhysicalDeviceMemoryProperties2KHR = vkGetPhysicalDeviceMemoryProperties2;
        vmaFuncs.vkGetDeviceBufferMemoryRequirements = vkGetDeviceBufferMemoryRequirements;
        vmaFuncs.vkGetDeviceImageMemoryRequirements = vkGetDeviceImageMemoryRequirements;

        VmaAllocatorCreateInfo allocatorInfo{};
        allocatorInfo.physicalDevice = physicalDevice;
        allocatorInfo.device = device;
        allocatorInfo.instance = instance;
        allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_2;
        allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
        allocatorInfo.pVulkanFunctions = &vmaFuncs;

        if (vmaCreateAllocator(&allocatorInfo, &allocator) != VK_SUCCESS) {
            throw std::runtime_error("vmaCreateAllocator failed");
        }

        VkQueue graphicsQueue = VK_NULL_HANDLE;
        VkQueue presentQueue = VK_NULL_HANDLE;
        vkGetDeviceQueue(device, queues.graphics, 0, &graphicsQueue);
        vkGetDeviceQueue(device, queues.present, 0, &presentQueue);

        const SwapchainSupport sc = querySwapchainSupport(physicalDevice, surface);
        if (sc.formats.empty() || sc.presentModes.empty()) {
            throw std::runtime_error("Swapchain support incomplete");
        }

        const VkSurfaceFormatKHR surfaceFormat = chooseSurfaceFormat(sc.formats);
        const VkPresentModeKHR presentMode = choosePresentMode(sc.presentModes);
        const VkExtent2D extent = chooseExtent(sc.capabilities, window);

        uint32_t imageCount = sc.capabilities.minImageCount + 1;
        if (sc.capabilities.maxImageCount > 0 && imageCount > sc.capabilities.maxImageCount) {
            imageCount = sc.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR scci{ VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
        scci.surface = surface;
        scci.minImageCount = imageCount;
        scci.imageFormat = surfaceFormat.format;
        scci.imageColorSpace = surfaceFormat.colorSpace;
        scci.imageExtent = extent;
        scci.imageArrayLayers = 1;
        scci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        const uint32_t qIndices[] = { queues.graphics, queues.present };
        if (queues.graphics != queues.present) {
            scci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            scci.queueFamilyIndexCount = 2;
            scci.pQueueFamilyIndices = qIndices;
        } else {
            scci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        scci.preTransform = sc.capabilities.currentTransform;
        scci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        scci.presentMode = presentMode;
        scci.clipped = VK_TRUE;

        VkSwapchainKHR swapchain = VK_NULL_HANDLE;
        if (vkCreateSwapchainKHR(device, &scci, nullptr, &swapchain) != VK_SUCCESS) {
            throw std::runtime_error("vkCreateSwapchainKHR failed");
        }

        uint32_t swapImageCount = 0;
        vkGetSwapchainImagesKHR(device, swapchain, &swapImageCount, nullptr);
        std::vector<VkImage> swapImages(swapImageCount);
        vkGetSwapchainImagesKHR(device, swapchain, &swapImageCount, swapImages.data());

        std::vector<VkImageView> swapViews(swapImageCount);
        for (uint32_t i = 0; i < swapImageCount; ++i) {
            VkImageViewCreateInfo ivci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            ivci.image = swapImages[i];
            ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            ivci.format = surfaceFormat.format;
            ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            ivci.subresourceRange.baseMipLevel = 0;
            ivci.subresourceRange.levelCount = 1;
            ivci.subresourceRange.baseArrayLayer = 0;
            ivci.subresourceRange.layerCount = 1;

            if (vkCreateImageView(device, &ivci, nullptr, &swapViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("vkCreateImageView failed");
            }
        }

        VkImage tracedImage = VK_NULL_HANDLE;
        VmaAllocation tracedImageAlloc = VK_NULL_HANDLE;
        VkImageView tracedImageView = VK_NULL_HANDLE;

        VkImage accumImage = VK_NULL_HANDLE;
        VmaAllocation accumImageAlloc = VK_NULL_HANDLE;
        VkImageView accumImageView = VK_NULL_HANDLE;

        auto createStorageImage = [&](VkImage& image, VmaAllocation& alloc, VkImageView& view, const char* debugName) {
            VkImageCreateInfo ici{};
            ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType = VK_IMAGE_TYPE_2D;
            ici.extent.width = extent.width;
            ici.extent.height = extent.height;
            ici.extent.depth = 1;
            ici.mipLevels = 1;
            ici.arrayLayers = 1;
            ici.format = VK_FORMAT_R32G32B32A32_SFLOAT;
            ici.tiling = VK_IMAGE_TILING_OPTIMAL;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            ici.samples = VK_SAMPLE_COUNT_1_BIT;
            ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

            if (vmaCreateImage(allocator, &ici, &aci, &image, &alloc, nullptr) != VK_SUCCESS) {
                throw std::runtime_error(std::string("vmaCreateImage failed for ") + debugName);
            }

            VkImageViewCreateInfo ivci{};
            ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            ivci.image = image;
            ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            ivci.format = VK_FORMAT_R32G32B32A32_SFLOAT;
            ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            ivci.subresourceRange.baseMipLevel = 0;
            ivci.subresourceRange.levelCount = 1;
            ivci.subresourceRange.baseArrayLayer = 0;
            ivci.subresourceRange.layerCount = 1;

            if (vkCreateImageView(device, &ivci, nullptr, &view) != VK_SUCCESS) {
                throw std::runtime_error(std::string("vkCreateImageView failed for ") + debugName);
            }
        };

        createStorageImage(tracedImage, tracedImageAlloc, tracedImageView, "tracedImage");
        createStorageImage(accumImage, accumImageAlloc, accumImageView, "accumImage");

        VkSampler presentSampler = VK_NULL_HANDLE;
        {
            VkSamplerCreateInfo sci{};
            sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sci.magFilter = VK_FILTER_LINEAR;
            sci.minFilter = VK_FILTER_LINEAR;
            sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.minLod = 0.0f;
            sci.maxLod = 1.0f;

            if (vkCreateSampler(device, &sci, nullptr, &presentSampler) != VK_SUCCESS) {
                throw std::runtime_error("vkCreateSampler failed");
            }
        }

        // Fullscreen quad via vertex shader generated positions; no vertex buffer needed.
        const auto vertCode = readSpvFile("Shared/fullscreen.vert.spv");
        const auto fragCode = readSpvFile("Shared/present.frag.spv");
        VkShaderModule vertModule = createShaderModule(device, vertCode);
        VkShaderModule fragModule = createShaderModule(device, fragCode);

        VkPipelineShaderStageCreateInfo shaderStages[2]{};
        shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        shaderStages[0].module = vertModule;
        shaderStages[0].pName = "main";
        shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        shaderStages[1].module = fragModule;
        shaderStages[1].pName = "main";

        VkDescriptorSetLayout computeSetLayout = VK_NULL_HANDLE;
        {
            std::array<VkDescriptorSetLayoutBinding, 11> bindings{};

            bindings[0].binding = 0;
            bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            bindings[0].descriptorCount = 1;
            bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            bindings[1].binding = 1;
            bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[1].descriptorCount = 1;
            bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            bindings[2].binding = 2;
            bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[2].descriptorCount = 1;
            bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            bindings[3].binding = 3;
            bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[3].descriptorCount = 1;
            bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            bindings[4].binding = 4;
            bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            bindings[4].descriptorCount = 1;
            bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            bindings[5].binding = 5;
            bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[5].descriptorCount = 1;
            bindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            bindings[6].binding = 6;
            bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[6].descriptorCount = 1;
            bindings[6].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            bindings[7].binding = 7;
            bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[7].descriptorCount = 1;
            bindings[7].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            bindings[8].binding = 8;
            bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[8].descriptorCount = 1;
            bindings[8].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            bindings[9].binding = 9;
            bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[9].descriptorCount = 1;
            bindings[9].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            bindings[10].binding = 10;
            bindings[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[10].descriptorCount = 1;
            bindings[10].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            VkDescriptorSetLayoutCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            ci.bindingCount = static_cast<uint32_t>(bindings.size());
            ci.pBindings = bindings.data();

            if (vkCreateDescriptorSetLayout(device, &ci, nullptr, &computeSetLayout) != VK_SUCCESS) {
                throw std::runtime_error("vkCreateDescriptorSetLayout(computeSetLayout) failed");
            }
        }

        VkDescriptorSetLayout presentSetLayout = VK_NULL_HANDLE;
        {
            std::array<VkDescriptorSetLayoutBinding, 1> bindings{};

            bindings[0].binding = 0;
            bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[0].descriptorCount = 1;
            bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

            VkDescriptorSetLayoutCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            ci.bindingCount = static_cast<uint32_t>(bindings.size());
            ci.pBindings = bindings.data();

            if (vkCreateDescriptorSetLayout(device, &ci, nullptr, &presentSetLayout) != VK_SUCCESS) {
                throw std::runtime_error("vkCreateDescriptorSetLayout(presentSetLayout) failed");
            }
        }

        VkPipelineLayout computePipelineLayout = VK_NULL_HANDLE;
        {
            VkPipelineLayoutCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            ci.setLayoutCount = 1;
            ci.pSetLayouts = &computeSetLayout;

            if (vkCreatePipelineLayout(device, &ci, nullptr, &computePipelineLayout) != VK_SUCCESS) {
                throw std::runtime_error("vkCreatePipelineLayout(compute) failed");
            }
        }

        VkPipelineLayout presentPipelineLayout = VK_NULL_HANDLE;
        {
            VkPipelineLayoutCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            ci.setLayoutCount = 1;
            ci.pSetLayouts = &presentSetLayout;

            if (vkCreatePipelineLayout(device, &ci, nullptr, &presentPipelineLayout) != VK_SUCCESS) {
                throw std::runtime_error("vkCreatePipelineLayout(present) failed");
            }
        }

        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = surfaceFormat.format;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorRef{};
        colorRef.attachment = 0;
        colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorRef;

        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo rpci{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
        rpci.attachmentCount = 1;
        rpci.pAttachments = &colorAttachment;
        rpci.subpassCount = 1;
        rpci.pSubpasses = &subpass;
        rpci.dependencyCount = 1;
        rpci.pDependencies = &dep;

        VkRenderPass renderPass = VK_NULL_HANDLE;
        if (vkCreateRenderPass(device, &rpci, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("vkCreateRenderPass failed");
        }

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        (void)io;
        ImGui::StyleColorsDark();

        ImGui_ImplGlfw_InitForVulkan(window, true);

        VkDescriptorPool imguiDescriptorPool = VK_NULL_HANDLE;
        {
            std::array<VkDescriptorPoolSize, 1> poolSizes{};
            poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            poolSizes[0].descriptorCount = 1024;

            VkDescriptorPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
            poolInfo.maxSets = 1024;
            poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
            poolInfo.pPoolSizes = poolSizes.data();

            if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &imguiDescriptorPool) != VK_SUCCESS) {
                throw std::runtime_error("vkCreateDescriptorPool(imguiDescriptorPool) failed");
            }
        }

        ImGui_ImplVulkan_InitInfo init_info{};
        init_info.ApiVersion = VK_API_VERSION_1_2;
        init_info.Instance = instance;
        init_info.PhysicalDevice = physicalDevice;
        init_info.Device = device;
        init_info.QueueFamily = queues.graphics;
        init_info.Queue = graphicsQueue;
        init_info.DescriptorPool = imguiDescriptorPool;
        init_info.MinImageCount = imageCount;
        init_info.ImageCount = swapImageCount;
        init_info.CheckVkResultFn = check_vk_result;

        init_info.PipelineInfoMain.RenderPass = renderPass;
        init_info.PipelineInfoMain.Subpass = 0;
        init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

        ImGui_ImplVulkan_Init(&init_info);

        std::vector<VkFramebuffer> framebuffers(swapImageCount);
        for (uint32_t i = 0; i < swapImageCount; ++i) {
            VkFramebufferCreateInfo fbci{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
            fbci.renderPass = renderPass;
            fbci.attachmentCount = 1;
            fbci.pAttachments = &swapViews[i];
            fbci.width = extent.width;
            fbci.height = extent.height;
            fbci.layers = 1;
            if (vkCreateFramebuffer(device, &fbci, nullptr, &framebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("vkCreateFramebuffer failed");
            }
        }

        VkPipelineVertexInputStateCreateInfo vertexInput{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(extent.width);
        viewport.height = static_cast<float>(extent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = extent;

        VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo raster{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo msaa{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
        msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo colorBlend{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
        colorBlend.attachmentCount = 1;
        colorBlend.pAttachments = &colorBlendAttachment;

        const auto compCode = readSpvFile("Shared/wavefront_rt.comp.spv");
        VkShaderModule compModule = createShaderModule(device, compCode);

        VkComputePipelineCreateInfo cpci2{};
        cpci2.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci2.layout = computePipelineLayout;
        cpci2.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci2.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci2.stage.module = compModule;
        cpci2.stage.pName = "main";

        VkPipeline computePipeline = VK_NULL_HANDLE;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci2, nullptr, &computePipeline) != VK_SUCCESS) {
            throw std::runtime_error("vkCreateComputePipelines failed");
        }

        VkGraphicsPipelineCreateInfo gpci{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
        gpci.stageCount = 2;
        gpci.pStages = shaderStages;
        gpci.pVertexInputState = &vertexInput;
        gpci.pInputAssemblyState = &inputAssembly;
        gpci.pViewportState = &viewportState;
        gpci.pRasterizationState = &raster;
        gpci.pMultisampleState = &msaa;
        gpci.pColorBlendState = &colorBlend;
        gpci.layout = presentPipelineLayout;
        gpci.renderPass = renderPass;
        gpci.subpass = 0;

        VkPipeline presentPipeline = VK_NULL_HANDLE;
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpci, nullptr, &presentPipeline) != VK_SUCCESS) {
            throw std::runtime_error("vkCreateGraphicsPipelines failed");
        }

        VkCommandPoolCreateInfo cpci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpci.queueFamilyIndex = queues.graphics;
        VkCommandPool commandPool = VK_NULL_HANDLE;
        if (vkCreateCommandPool(device, &cpci, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("vkCreateCommandPool failed");
        }

        {
            VkCommandBuffer fontCmd = VK_NULL_HANDLE;

            VkCommandBufferAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandPool = commandPool;
            allocInfo.commandBufferCount = 1;

            vkAllocateCommandBuffers(device, &allocInfo, &fontCmd);

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(fontCmd, &beginInfo);

            vkEndCommandBuffer(fontCmd);

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &fontCmd;

            vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(graphicsQueue);

            vkFreeCommandBuffers(device, commandPool, 1, &fontCmd);
        }

        std::vector<VkCommandBuffer> commandBuffers(swapImageCount);
        VkCommandBufferAllocateInfo cbai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cbai.commandPool = commandPool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
        if (vkAllocateCommandBuffers(device, &cbai, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("vkAllocateCommandBuffers failed");
        }

        RtAccelContext rtCtx{};
        rtCtx.device = device;
        rtCtx.allocator = allocator;
        rtCtx.commandPool = commandPool;
        rtCtx.graphicsQueue = graphicsQueue;
        rtCtx.scratchAlignment = asProps.minAccelerationStructureScratchOffsetAlignment;

        AllocatedBuffer pathBuffer = createBuffer(
            rtCtx,
            sizeof(GpuPathState) * extent.width * extent.height,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
        );

        AllocatedBuffer frameParamsBuffer = createBuffer(
            rtCtx,
            sizeof(FrameParams),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT
        );

        AllocatedBuffer materialBuffer = createBuffer(
            rtCtx,
            sizeof(Material) * mesh.materials.size(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
        );

        AllocatedBuffer triangleMaterialBuffer = createBuffer(
            rtCtx,
            sizeof(uint32_t) * mesh.triangleMaterialIndices.size(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
        );

        AllocatedBuffer emissiveTriBuffer = createBuffer(
            rtCtx,
            sizeof(EmissiveTriangle) * emissiveTriangles.size(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
        );

        if (!emissiveTriangles.empty()) {
            uploadToBuffer(
                rtCtx,
                emissiveTriangles.data(),
                sizeof(EmissiveTriangle) * emissiveTriangles.size(),
                emissiveTriBuffer
            );
        }

        uploadToBuffer(
            rtCtx,
            mesh.materials.data(),
            sizeof(Material) * mesh.materials.size(),
            materialBuffer
        );

        uploadToBuffer(
            rtCtx,
            mesh.triangleMaterialIndices.data(),
            sizeof(uint32_t) * mesh.triangleMaterialIndices.size(),
            triangleMaterialBuffer
        );

        std::vector<float> vertexNormals = buildVertexNormals(mesh);

        AllocatedBuffer normalBuffer = createBuffer(
            rtCtx,
            sizeof(float) * vertexNormals.size(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
        );

        uploadToBuffer(
            rtCtx,
            vertexNormals.data(),
            sizeof(float)* vertexNormals.size(),
            normalBuffer
        );

        CameraState camera{};
        camera.px = 0.0f;
        camera.py = 0.1f;
        camera.pz = -2.0f;
        camera.yaw = 0.0f;
        camera.pitch = 0.0f;
        camera.fovY = 45.0f * 3.1415926535f / 180.0f;

        int selectedMaterial = 0;
        RenderSettings renderSettings{};

        uint32_t accumulationFrameIndex = 0;

        std::vector<float> rtVertices;
        rtVertices.reserve(mesh.vertices.size() * 3);
        for (const auto& v : mesh.vertices) {
            rtVertices.push_back(v.x);
            rtVertices.push_back(v.y);
            rtVertices.push_back(v.z);
        }

        AllocatedBuffer blasVertexBuffer;
        AllocatedBuffer blasIndexBuffer;

        BlasInputGeometry blasGeom{};
        blasGeom.vertices = &rtVertices;
        blasGeom.indices = &mesh.indices;

        std::cout << "rtCtx.device      = " << rtCtx.device << "\n";
        std::cout << "rtCtx.commandPool = " << rtCtx.commandPool << "\n";
        std::cout << "rtCtx.graphicsQ   = " << rtCtx.graphicsQueue << "\n";
        std::cout << "vkAllocateCommandBuffers = " << (void*)vkAllocateCommandBuffers << "\n";

        AccelerationStructure blas = buildBLAS(rtCtx, blasGeom, blasVertexBuffer, blasIndexBuffer);
        AccelerationStructure tlas = buildTLAS(rtCtx, blas);

        std::cout << "BLAS built. Address: " << blas.deviceAddress << "\n";
        std::cout << "TLAS built. Address: " << tlas.deviceAddress << "\n";

        std::array<VkDescriptorPoolSize, 5> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = 1;

        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = 7;

        poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        poolSizes[2].descriptorCount = 2;

        poolSizes[3].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        poolSizes[3].descriptorCount = 1;

        poolSizes[4].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[4].descriptorCount = 1;

        VkDescriptorPoolCreateInfo poolCI{};
        poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolCI.maxSets = 2;
        poolCI.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolCI.pPoolSizes = poolSizes.data();

        VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
        if (vkCreateDescriptorPool(device, &poolCI, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("vkCreateDescriptorPool failed");
        }

        VkDescriptorSet computeSet = VK_NULL_HANDLE;
        {
            VkDescriptorSetAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool = descriptorPool;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &computeSetLayout;

            if (vkAllocateDescriptorSets(device, &ai, &computeSet) != VK_SUCCESS) {
                throw std::runtime_error("vkAllocateDescriptorSets(computeSet) failed");
            }

            VkDescriptorBufferInfo frameInfo{ frameParamsBuffer.buffer, 0, sizeof(FrameParams) };
            VkDescriptorBufferInfo pathInfo{ pathBuffer.buffer, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo vertexInfo{ blasVertexBuffer.buffer, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo indexInfo{ blasIndexBuffer.buffer, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo triMatInfo{ triangleMaterialBuffer.buffer, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo materialInfo{ materialBuffer.buffer, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo emissiveInfo{ emissiveTriBuffer.buffer, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo normalInfo{ normalBuffer.buffer, 0, VK_WHOLE_SIZE };

            VkDescriptorImageInfo storageImageInfo{};
            storageImageInfo.imageView = tracedImageView;
            storageImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkDescriptorImageInfo accumStorageImageInfo{};
            accumStorageImageInfo.imageView = accumImageView;
            accumStorageImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
            asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
            asInfo.accelerationStructureCount = 1;
            asInfo.pAccelerationStructures = &tlas.handle;

            std::array<VkWriteDescriptorSet, 11> writes{};

            writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[0].dstSet = computeSet;
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[0].pBufferInfo = &frameInfo;

            writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[1].dstSet = computeSet;
            writes[1].dstBinding = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[1].pBufferInfo = &pathInfo;

            writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[2].dstSet = computeSet;
            writes[2].dstBinding = 2;
            writes[2].descriptorCount = 1;
            writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[2].pImageInfo = &storageImageInfo; // tracedImage

            writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[3].dstSet = computeSet;
            writes[3].dstBinding = 3;
            writes[3].descriptorCount = 1;
            writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[3].pImageInfo = &accumStorageImageInfo; // accumImage

            writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[4].dstSet = computeSet;
            writes[4].dstBinding = 4;
            writes[4].descriptorCount = 1;
            writes[4].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            writes[4].pNext = &asInfo;

            writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[5].dstSet = computeSet;
            writes[5].dstBinding = 5;
            writes[5].descriptorCount = 1;
            writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[5].pBufferInfo = &vertexInfo;

            writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[6].dstSet = computeSet;
            writes[6].dstBinding = 6;
            writes[6].descriptorCount = 1;
            writes[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[6].pBufferInfo = &indexInfo;

            writes[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[7].dstSet = computeSet;
            writes[7].dstBinding = 7;
            writes[7].descriptorCount = 1;
            writes[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[7].pBufferInfo = &triMatInfo;

            writes[8] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[8].dstSet = computeSet;
            writes[8].dstBinding = 8;
            writes[8].descriptorCount = 1;
            writes[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[8].pBufferInfo = &materialInfo;

            writes[9] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[9].dstSet = computeSet;
            writes[9].dstBinding = 9;
            writes[9].descriptorCount = 1;
            writes[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[9].pBufferInfo = &emissiveInfo;

            writes[10] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[10].dstSet = computeSet;
            writes[10].dstBinding = 10;
            writes[10].descriptorCount = 1;
            writes[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[10].pBufferInfo = &normalInfo;

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }

        VkDescriptorSet presentSet = VK_NULL_HANDLE;
        {
            VkDescriptorSetAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool = descriptorPool;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &presentSetLayout;

            if (vkAllocateDescriptorSets(device, &ai, &presentSet) != VK_SUCCESS) {
                throw std::runtime_error("vkAllocateDescriptorSets(presentSet) failed");
            }

            VkDescriptorImageInfo sampledImageInfo{};
            sampledImageInfo.sampler = presentSampler;
            sampledImageInfo.imageView = accumImageView;
            sampledImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            write.dstSet = presentSet;
            write.dstBinding = 0;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.pImageInfo = &sampledImageInfo;

            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        }

        VkDescriptorBufferInfo vertexInfo{};
        vertexInfo.buffer = blasVertexBuffer.buffer;
        vertexInfo.offset = 0;
        vertexInfo.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo indexInfo{};
        indexInfo.buffer = blasIndexBuffer.buffer;
        indexInfo.offset = 0;
        indexInfo.range = VK_WHOLE_SIZE;

        VkSemaphoreCreateInfo sci{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        VkFenceCreateInfo fci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkSemaphore renderFinished = VK_NULL_HANDLE;
        VkFence inFlight = VK_NULL_HANDLE;
        vkCreateSemaphore(device, &sci, nullptr, &imageAvailable);
        vkCreateSemaphore(device, &sci, nullptr, &renderFinished);
        vkCreateFence(device, &fci, nullptr, &inFlight);

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            bool materialChanged = drawMaterialEditor(mesh, selectedMaterial);
            bool settingsChanged = drawRendererSettingsEditor(renderSettings);

            ImGui::Render();

            bool cameraChanged = updateCameraFromInput(window, camera);

            if (cameraChanged || materialChanged || settingsChanged || !renderSettings.accumulate) {
                accumulationFrameIndex = 0;
            }
            else {
                ++accumulationFrameIndex;
            }

            static size_t emissiveTriBufferCapacity = 0;

            if (materialChanged) {
                uploadToBuffer(
                    rtCtx,
                    mesh.materials.data(),
                    sizeof(Material) * mesh.materials.size(),
                    materialBuffer
                );

                emissiveTriangles = buildEmissiveTriangles(mesh);

                destroyBuffer(rtCtx, emissiveTriBuffer);

                emissiveTriBuffer = createBuffer(
                    rtCtx,
                    std::max<size_t>(sizeof(EmissiveTriangle), sizeof(EmissiveTriangle) * emissiveTriangles.size()),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
                );

                if (!emissiveTriangles.empty()) {
                    uploadToBuffer(
                        rtCtx,
                        emissiveTriangles.data(),
                        sizeof(EmissiveTriangle) * emissiveTriangles.size(),
                        emissiveTriBuffer
                    );
                }

                VkDescriptorBufferInfo emissiveInfo{};
                emissiveInfo.buffer = emissiveTriBuffer.buffer;
                emissiveInfo.offset = 0;
                emissiveInfo.range = VK_WHOLE_SIZE;

                VkWriteDescriptorSet write{};
                write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write.dstSet = computeSet;
                write.dstBinding = 9;
                write.descriptorCount = 1;
                write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                write.pBufferInfo = &emissiveInfo;

                vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

                accumulationFrameIndex = 0;
            }

            FrameParams fp{};
            fp.width = extent.width;
            fp.height = extent.height;
            fp.pathCount = extent.width * extent.height;
            fp.frameIndex = renderSettings.accumulate ? accumulationFrameIndex : 0u;

            fp.emissiveTriangleCount = static_cast<uint32_t>(emissiveTriangles.size());
            fp.sunEnabled = renderSettings.sunEnabled ? 1u : 0u;
            fp.fogEnabled = renderSettings.fogEnabled ? 1u : 0u;
            fp.maxBounces = static_cast<uint32_t>(renderSettings.maxBounces);

            float sx = renderSettings.sunDirection[0];
            float sy = renderSettings.sunDirection[1];
            float sz = renderSettings.sunDirection[2];
            normalize3(sx, sy, sz);

            fp.sunDirIntensity[0] = sx;
            fp.sunDirIntensity[1] = sy;
            fp.sunDirIntensity[2] = sz;
            fp.sunDirIntensity[3] = renderSettings.sunIntensity;

            fp.sunColor[0] = renderSettings.sunColor[0];
            fp.sunColor[1] = renderSettings.sunColor[1];
            fp.sunColor[2] = renderSettings.sunColor[2];
            fp.sunColor[3] = 0.0f;

            fp.fogParams[0] = renderSettings.fogDensity;
            fp.fogParams[1] = renderSettings.fogG;
            fp.fogParams[2] = 0.0f;
            fp.fogParams[3] = 0.0f;

            CameraUBO camUbo = makeCameraUBO(camera, extent.width, extent.height);
            std::memcpy(fp.camPos, camUbo.camPos, sizeof(camUbo.camPos));
            std::memcpy(fp.camForward, camUbo.camForward, sizeof(camUbo.camForward));
            std::memcpy(fp.camRight, camUbo.camRight, sizeof(camUbo.camRight));
            std::memcpy(fp.camUp, camUbo.camUp, sizeof(camUbo.camUp));
            std::memcpy(fp.camData, camUbo.camData, sizeof(camUbo.camData));

            std::memcpy(frameParamsBuffer.allocInfo.pMappedData, &fp, sizeof(fp));
            vmaFlushAllocation(allocator, frameParamsBuffer.allocation, 0, sizeof(fp));

            vkWaitForFences(device, 1, &inFlight, VK_TRUE, UINT64_MAX);
            vkResetFences(device, 1, &inFlight);

            uint32_t imageIndex = 0;
            VkResult acquire = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, imageAvailable, VK_NULL_HANDLE, &imageIndex);
            if (acquire != VK_SUCCESS) {
                throw std::runtime_error("vkAcquireNextImageKHR failed");
            }

            VkCommandBuffer cmd = commandBuffers[imageIndex];

            vkResetCommandBuffer(cmd, 0);

            VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) {
                throw std::runtime_error("vkBeginCommandBuffer failed");
            }

            VkImageMemoryBarrier tracedToGeneral{};
            tracedToGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            tracedToGeneral.oldLayout =
                (accumulationFrameIndex == 0)
                ? VK_IMAGE_LAYOUT_UNDEFINED
                : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            tracedToGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            tracedToGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            tracedToGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            tracedToGeneral.image = tracedImage;
            tracedToGeneral.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            tracedToGeneral.subresourceRange.baseMipLevel = 0;
            tracedToGeneral.subresourceRange.levelCount = 1;
            tracedToGeneral.subresourceRange.baseArrayLayer = 0;
            tracedToGeneral.subresourceRange.layerCount = 1;
            tracedToGeneral.srcAccessMask = 0;
            tracedToGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

            VkImageMemoryBarrier accumToGeneral{};
            accumToGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            accumToGeneral.oldLayout =
                (accumulationFrameIndex == 0)
                ? VK_IMAGE_LAYOUT_UNDEFINED
                : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            accumToGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            accumToGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            accumToGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            accumToGeneral.image = accumImage;
            accumToGeneral.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            accumToGeneral.subresourceRange.baseMipLevel = 0;
            accumToGeneral.subresourceRange.levelCount = 1;
            accumToGeneral.subresourceRange.baseArrayLayer = 0;
            accumToGeneral.subresourceRange.layerCount = 1;
            accumToGeneral.srcAccessMask = 0;
            accumToGeneral.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

            VkImageMemoryBarrier toGeneralBarriers[] = { tracedToGeneral, accumToGeneral };

            vkCmdPipelineBarrier(
                cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr,
                0, nullptr,
                2, toGeneralBarriers
            );

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
            vkCmdBindDescriptorSets(
                cmd,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                computePipelineLayout,
                0,
                1,
                &computeSet,
                0,
                nullptr
            );
            vkCmdDispatch(cmd, (extent.width + 7) / 8, (extent.height + 7) / 8, 1);

            VkImageMemoryBarrier tracedToSampled{};
            tracedToSampled.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            tracedToSampled.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            tracedToSampled.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            tracedToSampled.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            tracedToSampled.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            tracedToSampled.image = tracedImage;
            tracedToSampled.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            tracedToSampled.subresourceRange.baseMipLevel = 0;
            tracedToSampled.subresourceRange.levelCount = 1;
            tracedToSampled.subresourceRange.baseArrayLayer = 0;
            tracedToSampled.subresourceRange.layerCount = 1;
            tracedToSampled.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            tracedToSampled.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            VkImageMemoryBarrier accumToSampled{};
            accumToSampled.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            accumToSampled.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            accumToSampled.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            accumToSampled.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            accumToSampled.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            accumToSampled.image = accumImage;
            accumToSampled.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            accumToSampled.subresourceRange.baseMipLevel = 0;
            accumToSampled.subresourceRange.levelCount = 1;
            accumToSampled.subresourceRange.baseArrayLayer = 0;
            accumToSampled.subresourceRange.layerCount = 1;
            accumToSampled.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            accumToSampled.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            VkImageMemoryBarrier toSampledBarriers[] = { tracedToSampled, accumToSampled };

            vkCmdPipelineBarrier(
                cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0,
                0, nullptr,
                0, nullptr,
                2, toSampledBarriers
            );

            VkClearValue clear{};
            clear.color = { {0.02f, 0.02f, 0.02f, 1.0f} };

            VkRenderPassBeginInfo rpbi{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
            rpbi.renderPass = renderPass;
            rpbi.framebuffer = framebuffers[imageIndex];
            rpbi.renderArea.offset = { 0, 0 };
            rpbi.renderArea.extent = extent;
            rpbi.clearValueCount = 1;
            rpbi.pClearValues = &clear;

            vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, presentPipeline);
            vkCmdBindDescriptorSets(
                cmd,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                presentPipelineLayout,
                0,
                1,
                &presentSet,
                0,
                nullptr
            );
            vkCmdDraw(cmd, 3, 1, 0, 0);
            ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
            vkCmdEndRenderPass(cmd);

            if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
                throw std::runtime_error("vkEndCommandBuffer failed");
            }

            VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
            submit.waitSemaphoreCount = 1;
            submit.pWaitSemaphores = &imageAvailable;
            submit.pWaitDstStageMask = &waitStage;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &cmd;
            submit.signalSemaphoreCount = 1;
            submit.pSignalSemaphores = &renderFinished;

            if (vkQueueSubmit(graphicsQueue, 1, &submit, inFlight) != VK_SUCCESS) {
                throw std::runtime_error("vkQueueSubmit failed");
            }

            VkPresentInfoKHR present{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
            present.waitSemaphoreCount = 1;
            present.pWaitSemaphores = &renderFinished;
            present.swapchainCount = 1;
            present.pSwapchains = &swapchain;
            present.pImageIndices = &imageIndex;
            if (vkQueuePresentKHR(presentQueue, &present) != VK_SUCCESS) {
                throw std::runtime_error("vkQueuePresentKHR failed");
            }
        }

        vkDeviceWaitIdle(device);

        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        vkDestroyDescriptorPool(device, imguiPool, nullptr);

        vkDestroyFence(device, inFlight, nullptr);
        vkDestroySemaphore(device, renderFinished, nullptr);
        vkDestroySemaphore(device, imageAvailable, nullptr);
        destroyAccelerationStructure(rtCtx, tlas);
        destroyAccelerationStructure(rtCtx, blas);
        destroyBuffer(rtCtx, blasVertexBuffer);
        destroyBuffer(rtCtx, blasIndexBuffer);
        destroyBuffer(rtCtx, pathBuffer);
        destroyBuffer(rtCtx, frameParamsBuffer);

        vkDestroySampler(device, presentSampler, nullptr);
        vkDestroyImageView(device, tracedImageView, nullptr);
        vmaDestroyImage(allocator, tracedImage, tracedImageAlloc);
        vkDestroyImageView(device, accumImageView, nullptr);
        vmaDestroyImage(allocator, accumImage, accumImageAlloc);

        vkDestroyDescriptorSetLayout(device, computeSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, presentSetLayout, nullptr);

        vkDestroyPipeline(device, computePipeline, nullptr);
        vkDestroyPipeline(device, presentPipeline, nullptr);

        vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);
        vkDestroyPipelineLayout(device, presentPipelineLayout, nullptr);

        vkDestroyShaderModule(device, compModule, nullptr);
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vmaDestroyAllocator(allocator);
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);
        vkDestroyShaderModule(device, fragModule, nullptr);
        vkDestroyShaderModule(device, vertModule, nullptr);
        for (auto fb : framebuffers) vkDestroyFramebuffer(device, fb, nullptr);
        for (auto v : swapViews) vkDestroyImageView(device, v, nullptr);
        vkDestroySwapchainKHR(device, swapchain, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        glfwDestroyWindow(window);
        glfwTerminate();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
}

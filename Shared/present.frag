#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D renderTex;

void main() {
    vec3 c = texture(renderTex, inUV).rgb;
    outColor = vec4(c, 1.0);
}
#version 450

layout(location = 0) in vec2 UV;
layout(location = 0) out vec4 FragColor;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main() {
    FragColor = vec4(texture(tex, UV).rgb, 1.0);
}
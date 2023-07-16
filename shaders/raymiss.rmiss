#version 460

#extension GL_EXT_ray_tracing : require

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require
#include "ray_struct.inc.glsl"

layout(location = 0) rayPayloadInEXT Payload payload;

void main() {
    payload.hit_value = vec3(0.0, 0.0, 0.0);
}
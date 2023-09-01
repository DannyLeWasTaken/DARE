#version 460

#extension GL_EXT_ray_tracing : require

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require
#include "ray_struct.inc.glsl"

layout(location = 0) rayPayloadInEXT Payload payload;

void main() {
    //float a = 0.5*(payload.current.uv.y + 1.0);
    //payload.current.incoming_light += payload.current.hit_value *
    //((1.0 - a) * vec3(1.0) + a * vec3(0.5, 0.7, 1.0)) * 20.f;
    //payload.current.incoming_light = vec3(0.0);
    
    payload.current.missed = true;
}
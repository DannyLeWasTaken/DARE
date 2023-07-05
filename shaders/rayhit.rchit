#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference2 : require

struct Payload {
    vec3 hit_value;
    vec2 uv;
};

layout(location = 0) rayPayloadInEXT Payload payload;

hitAttributeEXT vec3 attribs;

void main() {
   payload.hit_value = vec3(payload.uv.xy, 0.0);
}
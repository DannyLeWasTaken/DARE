#include "ray_struct.inc.glsl"

// Apply a position offset to prevent self-intersections
void apply_position_offset(inout Ray ray, const vec3 world_normal, const float max_normal_bias) {
    const float min_bias = 5e-6f;
    const float max_bias = max(min_bias, max_normal_bias);
    const float normal_bias = mix(min_bias, max_bias, clamp(dot(world_normal, ray.ray_direction), 0.0, 1.0));
    ray.ray_origin += ray.ray_direction * normal_bias;
}

// Return true if a vector approaches to zero in all directions
bool near_zero(vec3 vec) {
    const float s = 1e-8;
    return (vec.x < s) && (vec.y < s) && (vec.z < s);
}
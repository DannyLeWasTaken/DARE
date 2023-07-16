#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "ray_struct.inc.glsl"

hitAttributeEXT vec2 attribs;

layout(location = 0) rayPayloadInEXT Payload payload;

layout(buffer_reference, scalar) buffer Vertices { vec3 v[]; }; // Vertices
layout(buffer_reference, scalar) buffer Normals { vec3 v[]; }; // Normals
layout(buffer_reference, scalar) buffer TexCoords { vec2 v[]; }; // TexCoords
layout(buffer_reference, scalar) buffer Indices {ivec3 i[]; }; // Triangle indices


layout(set = 1, binding = 1, scalar) buffer ObjectDescription_ {
    ObjectDescription i[];
} ObjectDescriptionArray;
layout (set = 1, binding = 2, scalar) buffer MaterialDescription_ {
    MaterialDescription m[];
} MaterialDescriptionArray;
layout(set = 1, binding = 3) uniform sampler2D texture_samplers[];

void main() {
    ObjectDescription object_resource = ObjectDescriptionArray.i[gl_InstanceCustomIndexEXT];
    MaterialDescription material_resource = MaterialDescriptionArray.m[object_resource.material];
    Vertices          vertices        = Vertices(object_resource.vertex_buffer);
    Normals           normals         = Normals(object_resource.normal_buffer);
    TexCoords         tex_coords      = TexCoords(object_resource.tex_buffer);
    Indices           indices         = Indices(object_resource.index_buffer);


    ivec3 ind = indices.i[gl_PrimitiveID];

    // Vertices of the triangle
    vec3 v0 = vertices.v[ind.x];
    vec3 v1 = vertices.v[ind.y];
    vec3 v2 = vertices.v[ind.z];

    // Tex coords of the triangle
    vec2 t_v0 = tex_coords.v[ind.x];
    vec2 t_v1 = tex_coords.v[ind.y];
    vec2 t_v2 = tex_coords.v[ind.z];
    const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    vec2 tex_coord = t_v0 * barycentrics.x + t_v1 * barycentrics.y + t_v2 * barycentrics.z;

    // Computing the coordinates of the hit position
    const vec3 pos      = v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
    const vec3 world_position = vec3(gl_ObjectToWorldEXT * vec4(pos, 1.0));  // Transforming the position to world space

    vec3 n_v0 = normals.v[ind.x];
    vec3 n_v1 = normals.v[ind.y];
    vec3 n_v2 = normals.v[ind.z];
    payload.hit_value = normalize(n_v0);

    //const sampler2D image = texture_samplers[nonuniformEXT(0)];
    if (material_resource.albedo_texture >= 0) {
        payload.hit_value = texture(texture_samplers[nonuniformEXT(material_resource.albedo_texture)], tex_coord).rgb;
    }
    //payload.hit_value = vec3(tex_coord, 0.0);
    /*
    if (object_resource.tex_buffer != 0 && material_resource.albedo_texture >= 0) {
        vec2 texCoord = t_v0 * barycentrics.x + t_v1 * barycentrics.y + t_v2 * barycentrics.z;
        payload.hit_value = texture(texture_samplers[nonuniformEXT(material_resource.albedo_texture)], texCoord).xyz;
    }
    */
}
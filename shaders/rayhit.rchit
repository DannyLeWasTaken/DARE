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
//layout(buffer_reference, scalar) buffer ObjectDescriptionArray {ObjectDescription i[]; }; // Triangle indices
//layout(buffer_reference, scalar) buffer MaterialDescriptionArray {MaterialDescription i[]; }; // Triangle indices

layout(set = 1, binding = 1, scalar) buffer ObjectDescription_ {
    ObjectDescription i[];
} ObjectDescriptionArray;
layout (set = 1, binding = 2, scalar) buffer MaterialDescription_ {
    MaterialDescription m[];
} MaterialDescriptionArray;
layout(set = 1, binding = 3) uniform sampler2D texture_samplers[];

vec3 hsv_to_rgb(in vec3 hsv) {
    const vec3 rgb = clamp(abs(mod(hsv.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, vec3(0.0), vec3(1.0));
    return hsv.z * mix(vec3(1.0), rgb, hsv.y);
}

const float M_GOLDEN_CONJ = 0.6180339887498948482045868343656381177203091798057628621354486227;

void main() {
    //ObjectDescriptionArray   object_descriptions = ObjectDescriptionArray(DescriptionAddressesArray.i[0].object);
    //MaterialDescriptionArray material_descriptions = ObjectDescriptionArray(DescriptionAddressesArray.i[0].material);
    ObjectDescription object_resource = ObjectDescriptionArray.i[gl_InstanceCustomIndexEXT];
    if (object_resource.material > -1) {
        MaterialDescription material_resource = MaterialDescriptionArray.m[object_resource.material];
        Vertices vertices = Vertices(object_resource.vertex_buffer);
        Normals normals = Normals(object_resource.normal_buffer);
        TexCoords tex_coords = TexCoords(object_resource.tex_buffer);
        Indices indices = Indices(object_resource.index_buffer);

        ivec3 ind = indices.i[gl_PrimitiveID];
        const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

        // Triangle positions
        vec3 v0 = vertices.v[ind.x];
        vec3 v1 = vertices.v[ind.y];
        vec3 v2 = vertices.v[ind.z];
        const vec3 position = v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
        const vec3 world_position = vec3(gl_ObjectToWorldEXT * vec4(position, 1.0));  // Transforming the position to world space

        // Normal
        vec3 normal;
        vec3 world_normal;
        const vec3 geom_normal = normalize(cross(v0 - v1, v2 - v0));
        if (object_resource.normal_buffer > 0) {
            vec3 n_v0 = normals.v[ind.x];
            vec3 n_v1 = normals.v[ind.y];
            vec3 n_v2 = normals.v[ind.z];
            normal = normalize(n_v0 * barycentrics.x + n_v1 * barycentrics.y + n_v2 * barycentrics.z);
            world_normal = normalize(vec3(normal * gl_WorldToObjectEXT));
        } else {
            normal = geom_normal;
            world_normal = normalize(vec3(normal * gl_WorldToObjectEXT));
        }



        //const sampler2D image = texture_samplers[nonuniformEXT(0)];
        //payload.hit_value = vec3(0);

        if (material_resource.albedo_texture > -1 && object_resource.tex_buffer > 0) {
            // Tex coords of the triangle
            vec2 t_v0 = tex_coords.v[ind.x];
            vec2 t_v1 = tex_coords.v[ind.y];
            vec2 t_v2 = tex_coords.v[ind.z];
            vec2 tex_coord_0 = vec2(t_v0.x * barycentrics.x + t_v1.x * barycentrics.y + t_v2.x * barycentrics.z,
                                    1.0 - (t_v0.y * barycentrics.x + t_v1.y * barycentrics.y + t_v2.y * barycentrics.z));
            tex_coord_0.y = 1 - tex_coord_0.y; // Flip Y


            payload.hit_value = texture(texture_samplers[nonuniformEXT(material_resource.albedo_texture)], tex_coord_0).rgb
            * material_resource.albedo;
            //payload.hit_value = hsv_to_rgb(vec3(float(gl_InstanceCustomIndexEXT) * M_GOLDEN_CONJ, 0.875, 0.85));
            //payload.hit_value = hsv_to_rgb(vec3(float(material_resource.albedo_texture) * M_GOLDEN_CONJ, 0.875, 0.85));
            //payload.hit_value = vec3(tex_coord_0, 0);
        } else {
            payload.hit_value = material_resource.albedo;
            //payload.hit_value = hsv_to_rgb(vec3(float(gl_InstanceCustomIndexEXT) * M_GOLDEN_CONJ, 0.875, 0.85));
        }
    } else {
        //payload.hit_value = hsv_to_rgb(vec3(float(object_resource.material) * M_GOLDEN_CONJ, 0.875, 0.85));
        payload.hit_value = vec3(0.1, 0.1, 0.1);
    }
}
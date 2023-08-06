#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "ray_struct.inc.glsl"
#include "sampler.inc.glsl"

hitAttributeEXT vec2 attribs;

layout(location = 0) rayPayloadInEXT Payload payload;

layout(buffer_reference, scalar) buffer Vertices { vec3 v[]; }; // Vertices
layout(buffer_reference, scalar) buffer Normals { vec3 v[]; }; // Normals
layout(buffer_reference, scalar) buffer TexCoords { vec2 v[]; }; // TexCoords
layout(buffer_reference, scalar) buffer Indices { uvec3 i[]; }; // Triangle indices
//layout(buffer_reference, scalar) buffer ObjectDescriptionArray {ObjectDescription i[]; }; // Triangle indices
//layout(buffer_reference, scalar) buffer MaterialDescriptionArray {MaterialDescription i[]; }; // Triangle indices

layout(set = 1, binding = 1, scalar) buffer ObjectDescription_ {
    ObjectDescription i[];
} ObjectDescriptionArray;
layout (set = 1, binding = 2, scalar) buffer MaterialDescription_ {
    MaterialDescription m[];
} MaterialDescriptionArray;
layout(set = 1, binding = 3) uniform sampler2D texture_samplers[];

dvec3 hsv_to_rgb(in dvec3 hsv) {
    const dvec3 rgb = clamp(abs(mod(hsv.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, vec3(0.0), vec3(1.0));
    return hsv.z * mix(dvec3(1.0), rgb, hsv.y);
}

const float M_GOLDEN_CONJ = 0.6180339887498948482045868343656381177203091798057628621354486227;

void main() {
    //ObjectDescriptionArray   object_descriptions = ObjectDescriptionArray(DescriptionAddressesArray.i[0].object);
    //MaterialDescriptionArray material_descriptions = ObjectDescriptionArray(DescriptionAddressesArray.i[0].material);
    ObjectDescription object_resource = ObjectDescriptionArray.i[gl_InstanceCustomIndexEXT];
    if (payload.missed == true) {
        return;
    }

    if (object_resource.material > -1) {
        MaterialDescription material_resource = MaterialDescriptionArray.m[object_resource.material];
        Vertices vertices = Vertices(object_resource.vertex_buffer);
        Normals normals = Normals(object_resource.normal_buffer);
        TexCoords tex_coords = TexCoords(object_resource.tex_buffer);
        Indices indices = Indices(object_resource.index_buffer);

        uvec3 ind = indices.i[gl_PrimitiveID];
        const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

        // Triangle positions
        vec3 v0 = vertices.v[ind.x];
        vec3 v1 = vertices.v[ind.y];
        vec3 v2 = vertices.v[ind.z];
        const vec3 position = v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
        //const vec3 world_position = vec3(gl_ObjectToWorldEXT * vec4(position, 1.0));  // Transforming the position to world space
        const vec3 world_position = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

        // Normal
        vec3 normal;
        vec3 world_normal;
        const vec3 geometry_normal = normalize(cross(v0 - v1, v2 - v0));
        if (object_resource.tex_buffer > 0 && material_resource.normal.w > -1) {
            // Tex coords of the triangle
            vec2 t_v0 = tex_coords.v[ind.x];
            vec2 t_v1 = tex_coords.v[ind.y];
            vec2 t_v2 = tex_coords.v[ind.z];
            vec2 tex_coords = t_v0 * barycentrics.x + t_v1 * barycentrics.y + t_v2 * barycentrics.z;
            normal = normalize(texture(texture_samplers[nonuniformEXT(int(material_resource.albedo.w))], tex_coords).rgb);
        } else if (object_resource.normal_buffer > 0) {
            vec3 n_v0 = normals.v[ind.x];
            vec3 n_v1 = normals.v[ind.y];
            vec3 n_v2 = normals.v[ind.z];
            normal = normalize(n_v0 * barycentrics.x + n_v1 * barycentrics.y + n_v2 * barycentrics.z);
        } else {
            normal = geometry_normal;
        }
        world_normal = normalize(vec3(normal * gl_WorldToObjectEXT));



        //const sampler2D image = texture_samplers[nonuniformEXT(0)];
        //payload.hit_value = vec3(0);
        if (material_resource.albedo.w > -1 && object_resource.tex_buffer > 0 ) {
            // Tex coords of the triangle
            vec2 t_v0 = tex_coords.v[ind.x];
            vec2 t_v1 = tex_coords.v[ind.y];
            vec2 t_v2 = tex_coords.v[ind.z];
            //t_v0.y = 1.0 - t_v0.y;
            //t_v1.y = 1.0 - t_v1.y;
            //t_v2.y = 1.0 - t_v2.y;
            vec2 tex_coords = t_v0 * barycentrics.x + t_v1 * barycentrics.y + t_v2 * barycentrics.z;
            payload.hit_value *= texture(texture_samplers[nonuniformEXT(int(material_resource.albedo.w))], tex_coords).rgb
            * material_resource.albedo.xyz;

            if (material_resource.emissive.w > -1) {
                payload.incoming_light += payload.hit_value * texture(texture_samplers[nonuniformEXT(int(material_resource.emissive.w))], tex_coords).rgb * material_resource.emissive.rgb;
            } else {
                payload.incoming_light += payload.hit_value * material_resource.emissive.rgb;
            }

            //payload.hit_value = vec3(tex_coords, 0);
            //payload.hit_value = vec3(0, attribs.x, attribs.y);
            //payload.hit_value = vec3(hsv_to_rgb(dvec3(double(object_resource.normal_buffer) * double(M_GOLDEN_CONJ), 0.875, 0.85)));
            //payload.hit_value = vec3(hsv_to_rgb(dvec3(double(object_resource.tex_buffer) * double(M_GOLDEN_CONJ), 0.875, 0.85)));
            //payload.hit_value = hsv_to_rgb(vec3(float(gl_InstanceCustomIndexEXT) * M_GOLDEN_CONJ, 0.875, 0.85));
            //payload.hit_value = vec3(hsv_to_rgb(vec3(double(material_resource.albedo_texture) * double(M_GOLDEN_CONJ), 0.875, 0.85)));
            //payload.hit_value = vec3(tex_coord_0, 0);
        } else {
            payload.hit_value *= material_resource.albedo.xyz;
            payload.incoming_light += material_resource.emissive.xyz * payload.hit_value;
            //payload.hit_value = hsv_to_rgb(vec3(float(gl_InstanceCustomIndexEXT) * M_GOLDEN_CONJ, 0.875, 0.85));
        }
        payload.ray_origin = world_position;

        vec3 tangent, bitangent;
        createCoordinateSystem(world_normal, tangent, bitangent);
        vec3 ray_direction = samplingHemisphereUniform(payload.seed, tangent, bitangent, world_normal);
        payload.ray_direction = ray_direction;
    }
    payload.seed += 1;
    payload.bounces += 1;
}
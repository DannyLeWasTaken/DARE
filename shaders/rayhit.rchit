#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "ray_struct.inc.glsl"
#include "sampler.inc.glsl"
#include "lighting.inc.glsl"

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
const float EPSILON = 1e-4;
#define M_PI 3.14159265

void main() {
    //ObjectDescriptionArray   object_descriptions = ObjectDescriptionArray(DescriptionAddressesArray.i[0].object);
    //MaterialDescriptionArray material_descriptions = ObjectDescriptionArray(DescriptionAddressesArray.i[0].material);
    ObjectDescription object_resource = ObjectDescriptionArray.i[gl_InstanceCustomIndexEXT];
    if (payload.current.missed == true) {
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
        const vec3 world_position = vec3(gl_ObjectToWorldEXT * vec4(position, 1.0));  // Transforming the position to world space
        //const vec3 world_position = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

        // Normal
        vec3 normal;
        vec3 world_normal;
        vec3 geometry_normal = normalize(cross(v0 - v1, v2 - v0));
        vec3 world_geometry_normal = normalize(vec3(geometry_normal * gl_WorldToObjectEXT));
        if (object_resource.tex_buffer > 0 && material_resource.normal.w > -1) {
            // Tex coords of the triangle
            vec2 t_v0 = tex_coords.v[ind.x];
            vec2 t_v1 = tex_coords.v[ind.y];
            vec2 t_v2 = tex_coords.v[ind.z];
            vec2 tex_coords = t_v0 * barycentrics.x + t_v1 * barycentrics.y + t_v2 * barycentrics.z;
            normal = texture(texture_samplers[nonuniformEXT(int(material_resource.normal.w))], tex_coords).rgb;
        } else if (object_resource.normal_buffer > 0) {
            vec3 n_v0 = normals.v[ind.x];
            vec3 n_v1 = normals.v[ind.y];
            vec3 n_v2 = normals.v[ind.z];
            normal = normalize(n_v0 * barycentrics.x + n_v1 * barycentrics.y + n_v2 * barycentrics.z);
        } else {
            normal = geometry_normal;
        }
        world_normal = normalize(vec3(normal * gl_WorldToObjectEXT));

        // Back-face processing
        if (dot(gl_WorldRayDirectionEXT, -world_geometry_normal) > 0.0) {
            world_normal = -world_normal;
            normal = -normal;
            geometry_normal = -geometry_normal;
            world_geometry_normal = -world_geometry_normal;
        }

        vec3 final_color = payload.current.incoming_light;
        vec3 hit_value = payload.current.hit_value;
        vec3 albedo = vec3(1.0);

        // Albedo
        if (material_resource.albedo.w > -1 && object_resource.tex_buffer > 0 ) {
            // Tex coords of the triangle
            vec2 t_v0 = tex_coords.v[ind.x];
            vec2 t_v1 = tex_coords.v[ind.y];
            vec2 t_v2 = tex_coords.v[ind.z];
            vec2 tex_coords = t_v0 * barycentrics.x + t_v1 * barycentrics.y + t_v2 * barycentrics.z;
            albedo = texture(texture_samplers[nonuniformEXT(int(material_resource.albedo.w))], tex_coords).rgb
            * material_resource.albedo.rgb;
        } else {
            albedo = material_resource.albedo.rgb;
        }

        // Specular-glossiness
        vec3 specular = vec3(1.0);
        float glossiness = 1.0;
        if (material_resource.specular_glossiness_factor.r > -1) {
            vec2 t_v0 = tex_coords.v[ind.x];
            vec2 t_v1 = tex_coords.v[ind.y];
            vec2 t_v2 = tex_coords.v[ind.z];
            vec2 tex_coords = t_v0 * barycentrics.x + t_v1 * barycentrics.y + t_v2 * barycentrics.z;

            vec4 specular = texture(texture_samplers[nonuniformEXT(int(material_resource.specular_glossiness_diffuse_texture.r))], tex_coords).rgba;
            //specular = specular_glossiness.rgb * material_resource.specular_glossiness_factor.rgb;
            //glossiness = specular_glossiness.a * material_resource.specular_glossiness_factor.a;
        } else {
            specular = material_resource.specular_glossiness_factor.rgb;
            glossiness = material_resource.specular_glossiness_factor.a;
        }
        
        // Metallic-roughness
        float metallic; // b-channel
        float roughness; // g-channel
        if (material_resource.metallic_roughness.r > -1) {
            vec2 t_v0 = tex_coords.v[ind.x];
            vec2 t_v1 = tex_coords.v[ind.y];
            vec2 t_v2 = tex_coords.v[ind.z];
            vec2 tex_coords = t_v0 * barycentrics.x + t_v1 * barycentrics.y + t_v2 * barycentrics.z;

            vec4 metallic_roughness = texture(texture_samplers[nonuniformEXT(int(material_resource.metallic_roughness.r))], tex_coords).rgba;
            roughness = metallic_roughness.g * material_resource.metallic_roughness.g;
            metallic = metallic_roughness.b * material_resource.metallic_roughness.b;
        } else {
            roughness = material_resource.metallic_roughness.g;
            metallic = material_resource.metallic_roughness.b;
        }

        // Emissiveness
        vec3 emitted_light = vec3(0.0);
        hit_value *= albedo;
        if (material_resource.emissive.w > -1 && object_resource.tex_buffer > 0) {
            vec2 t_v0 = tex_coords.v[ind.x];
            vec2 t_v1 = tex_coords.v[ind.y];
            vec2 t_v2 = tex_coords.v[ind.z];
            vec2 tex_coords = t_v0 * barycentrics.x + t_v1 * barycentrics.y + t_v2 * barycentrics.z;
            emitted_light = texture(texture_samplers[nonuniformEXT(int(material_resource.emissive.w))], tex_coords).rgb * material_resource.emissive.rgb;
        } else {
            emitted_light = material_resource.emissive.rgb;
        }
        emitted_light *= 20.f;

        // Compute reflection
        vec3 incoming_direction = normalize(payload.current.ray_direction);
        vec3 reflected_direction = reflect(incoming_direction, world_normal);

        vec3 tangent, bitangent;
        createCoordinateSystem(world_normal, tangent, bitangent);

        vec3 hemisphere_direction = samplingHemisphereUniform(
            payload.current.seed,
            tangent,
            bitangent,
            world_normal
        );
        vec3 ray_direction = mix(reflected_direction, hemisphere_direction, roughness);
        // Frensel-slick approximation
        float cos_theta = clamp(dot(incoming_direction, world_normal), 0.0, 1.0);
        vec3 F_0 = mix(vec3(0.04), albedo, metallic);
        vec3 frensel = F_0 + (1.0 - F_0) * pow(1.0 - cos_theta, 5.0);

        // Blend the reflection based on the frensel term and add it to the final color
        vec3 reflection_contribution = frensel * payload.current.incoming_light;
        final_color += (emitted_light == vec3(0.f) ? hit_value : (albedo == vec3(0.f) ? (hit_value/albedo) : hit_value)) * (reflection_contribution + (emitted_light * metallic));

        payload.current.incoming_light = final_color;
        payload.current.hit_value = hit_value;
        payload.current.ray_origin = world_position;

        // Update the payload with the reflect direction
        payload.current.ray_direction = normalize(ray_direction);
    }
    payload.current.bounces += 1;
}
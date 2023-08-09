struct Ray {
    vec3 ray_direction;
    vec3 ray_origin;
    vec2 uv;
    uint seed;

    vec3 hit_value;
    vec3 incoming_light;
    bool missed;

    uint depth; // # of iterations the current ray has gone through
    uint bounces; // # of bounces (in this case are not reflections)
    uint reflections; // # of reflections
};

struct Payload {
    Ray current;
    Ray previous;
};

/// Describes the material properties of the object
struct MaterialDescription {
    // -1 - None
    // X,Y,Z -> Factor (optionally used)
    // W -> Texture index -> -1 = None
    vec4 albedo;
    vec4 normal; // x,y,z unused. w is texture index
    vec4 emissive;
    vec4 diffuse_factor;
    vec4 specular_glossiness_factor; // rgb -> glossiness, a -> specular
    vec4 specular_glossiness_diffuse_texture; // r -> specular glossiness texture, b -> diffuse texture
    vec4 metallic_roughness; // r -> metallic-roughness texture, g -> roughness, b -> metallness
};

/// Represents the values of a material description
struct MaterialValues {
    vec3 albedo;
    vec3 normal;
    vec3 emissive;
    vec4 diffuse;
    vec3 specular;
    float glossiness;
};

void get_material_values(uint64_t tex_coords, MaterialDescription material) {

};

/// Describes the object's bda buffers and materials
struct ObjectDescription {
    // 0 - None
    uint64_t vertex_buffer;
    uint64_t index_buffer;
    uint64_t normal_buffer;
    uint64_t tex_buffer;
    int material;
};

// The BDAs of the descriptions
struct DescriptionAddresses {
    uint64_t object;
    uint64_t material;
};
struct Payload {
    vec3 ray_direction;
    vec3 ray_origin;

    vec3 hit_value;
    vec2 uv;
    uint depth; // # of iterations the current ray has gone through
    uint bounces; // # of bounces (in this case are not reflections)
    uint reflections; // # of reflections
};

/// Describes the material properties of the object
struct MaterialDescription {
    // -1 - None
    int albedo_texture;
    vec3 albedo;
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
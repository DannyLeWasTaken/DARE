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
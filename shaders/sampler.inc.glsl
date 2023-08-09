#include "random.inc.glsl"

// Randomly samples from a cosine-weighted hemisphere oriented in the `z` direction.
// From Ray Tracing Gems section 16.6.1, "Cosine-Weighted Hemisphere Oriented to the Z-Axis"
vec3 samplingHemisphere(inout uint seed, in vec3 x, in vec3 y, in vec3 z)
{
    #define M_PI 3.14159265

    float r1 = rnd(seed);
    float r2 = rnd(seed);
    float sq = sqrt(r1);

    vec3 direction = vec3(cos(2 * M_PI * r2) * sq, sin(2 * M_PI * r2) * sq, sqrt(1. - r1));
    direction      = direction.x * x + direction.y * y + direction.z * z;

    return direction;
}

vec3 samplingHemisphereUniform(inout uint seed, in vec3 x, in vec3 y, in vec3 z)
{
    #define M_PI 3.14159265

    float u = rnd(seed); // uniform random number in [0, 1]
    float v = rnd(seed); // uniform random number in [0, 1]

    float theta = 2.0 * M_PI * u; // azimuthal angle
    float phi = acos(2.0 * v - 1.0); // polar angle

    // spherical to cartesian conversion
    float sinPhi = sin(phi);
    vec3 direction = vec3(sinPhi * cos(theta), sinPhi * sin(theta), cos(phi));

    // transform direction from z-oriented basis to world space
    direction = direction.x * x + direction.y * y + direction.z * z;

    return direction;
}

// Return the tangent and binormal from the incoming normal
void createCoordinateSystem(in vec3 N, out vec3 Nt, out vec3 Nb)
{
    if(abs(N.x) > abs(N.y))
    Nt = vec3(N.z, 0, -N.x) / sqrt(N.x * N.x + N.z * N.z);
    else
    Nt = vec3(0, -N.z, N.y) / sqrt(N.y * N.y + N.z * N.z);
    Nb = cross(N, Nt);
}
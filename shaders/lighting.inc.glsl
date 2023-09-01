vec3 frensel_shlick(float cos_theta, vec3 F_0) {
    return F_0 + (1.0 - F_0) * pow(1.0 - cos_theta, 5.0);
}

float ggx_distribution(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;
    float denominator = NdotH2 * (a2 - 1.0) + 1.0;
    if (denominator < 0.001) {
        return 0;
    } else {
        return a2 / (3.14159265 * denominator * denominator);
    }
}

float ggx_geometry(float NdotV, float NdotL, float roughness) {
    float k = roughness * roughness / 2.0;
    float geoV = NdotV / (NdotV * (1.0 - k) + k);
    float geoL = NdotL / (NdotL * (1.0 - k) + k);
    return geoV * geoL;
}

vec3 specular_brdf_ggx(vec3 V, vec3 L, vec3 N, vec3 F0, float roughness) {
    vec3 H = V + L;
    if (length(H) < 0.0001) {
        H = N;
    } else {
        H = normalize(H);
    }

    float NdotV = max(dot(N, V), 0.001); // Adding small value to prevent 0
    float NdotL = max(dot(N, L), 0.001); // Adding small value to prevent 0
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    vec3 F = frensel_shlick(VdotH, F0);
    float D = ggx_distribution(NdotH, roughness);
    float G = ggx_geometry(NdotV, NdotL, roughness);

    return (D * G * F) / (4.0 * NdotV * NdotL);
}

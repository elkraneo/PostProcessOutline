//
//  OutlineShaders.metal
//  PostProcessOutline
//

#include <metal_stdlib>
using namespace metal;

// MARK: - Pass 1: Mask

// Renders the selected mesh into a single-channel (R8) texture.
// Every fragment that passes depth writes white — building a silhouette.

struct MaskUniforms {
    float4x4 mvp;
};

vertex float4 outlineMaskVertex(
    const device packed_float3* positions [[buffer(0)]],
    constant MaskUniforms& u             [[buffer(1)]],
    uint vid                             [[vertex_id]]
) {
    return u.mvp * float4(float3(positions[vid]), 1.0);
}

fragment float4 outlineMaskFragment(float4 position [[position]]) {
    return float4(1.0);
}

// MARK: - Pass 2: Dilate

// Expands the mask by `radius` pixels to form the outline ring.
// Interior pixels (already white) are cleared so the ring only
// appears around the silhouette edge, not over the mesh surface.

kernel void outlineDilate(
    texture2d<float, access::read>  maskTex  [[texture(0)]],
    texture2d<float, access::write> edgeTex  [[texture(1)]],
    constant int32_t&               radius   [[buffer(0)]],
    uint2 gid                                [[thread_position_in_grid]]
) {
    const uint w = maskTex.get_width();
    const uint h = maskTex.get_height();
    if (gid.x >= w || gid.y >= h) return;

    // Interior pixels don't form part of the outline.
    if (maskTex.read(gid).r > 0.5) {
        edgeTex.write(float4(0.0), gid);
        return;
    }

    // Is any neighboring pixel (within a circular radius) inside the mask?
    bool nearMask = false;
    const int r = radius;
    for (int dy = -r; dy <= r && !nearMask; ++dy) {
        for (int dx = -r; dx <= r && !nearMask; ++dx) {
            if (dx * dx + dy * dy > r * r) continue;
            const int nx = int(gid.x) + dx;
            const int ny = int(gid.y) + dy;
            if (nx < 0 || ny < 0 || nx >= int(w) || ny >= int(h)) continue;
            if (maskTex.read(uint2(nx, ny)).r > 0.5) nearMask = true;
        }
    }

    edgeTex.write(float4(nearMask ? 1.0 : 0.0), gid);
}

// MARK: - Pass 3: Composite

// Blends the outline color over the source frame wherever the edge mask is set.

kernel void outlineComposite(
    texture2d<float, access::read>  sourceColor  [[texture(0)]],
    texture2d<float, access::read>  edgeMask     [[texture(1)]],
    texture2d<float, access::write> targetColor  [[texture(2)]],
    constant float4&                outlineColor [[buffer(0)]],
    uint2 gid                                    [[thread_position_in_grid]]
) {
    const uint w = sourceColor.get_width();
    const uint h = sourceColor.get_height();
    if (gid.x >= w || gid.y >= h) return;

    float4 color = sourceColor.read(gid);
    if (edgeMask.read(gid).r > 0.5) {
        color = mix(color, float4(outlineColor.rgb, 1.0), outlineColor.a);
    }
    targetColor.write(color, gid);
}

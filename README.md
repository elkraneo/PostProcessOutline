# PostProcessOutline

A RealityKit post-process effect that renders pixel-exact selection outlines around 3D objects using Metal.

![Outline Demo](https://elkraneo.com/content/images/2026/04/steps.gif)

> **Read the full write-up:** [RealityKit Outline Selection](https://elkraneo.com/realitykit-outline-selection/)

## Overview

This project demonstrates a **post-process outline technique** for RealityKit that operates entirely in screen-space. Unlike traditional approaches (bounding boxes or inverted hulls), this method:

- Never touches the scene hierarchy
- Works at screen resolution regardless of mesh complexity
- Produces outlines of consistent pixel width
- Handles occlusion correctly

## How It Works

The outline is rendered in **three passes** using `PostProcessEffect` (iOS 26+, macOS 26+):

### Pass 1: Silhouette Mask
Renders the selected mesh into a single-channel `R8Unorm` texture. Every pixel covered by the mesh becomes white.

```metal
vertex float4 outlineMaskVertex(
    const device packed_float3* positions [[buffer(0)]],
    constant MaskUniforms& u [[buffer(1)]],
    uint vid [[vertex_id]]
) {
    return u.mvp * float4(float3(positions[vid]), 1.0);
}

fragment float4 outlineMaskFragment(float4 position [[position]]) {
    return float4(1.0);
}
```

### Pass 2: Dilation
Expands the mask outward by `radius` pixels. Pixels outside the mesh but within range of the silhouette edge become white. Interior pixels are explicitly suppressed to zero, creating a ring at the boundary only.

```metal
kernel void outlineDilate(
    texture2d<float, access::read>  maskTex [[texture(0)]],
    texture2d<float, access::write> edgeTex [[texture(1)]],
    constant int32_t& radius [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
)
```

### Pass 3: Composite
Blends the outline color over the source frame wherever the edge ring is set.

```metal
kernel void outlineComposite(
    texture2d<float, access::read>  sourceColor [[texture(0)]],
    texture2d<float, access::read>  edgeMask [[texture(1)]],
    texture2d<float, access::write> targetColor [[texture(2)]],
    constant float4& outlineColor [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
)
```

## Requirements

- iOS 26.0+ / macOS 26.0+
- Xcode 26.0+
- Swift 6

> Note: `PostProcessEffect` is unavailable on visionOS.

## Usage

```swift
import RealityKit
import SwiftUI

@available(macOS 26.0, iOS 26.0, *)
struct ContentView: View {
    @State private var effect = PostProcessOutlineEffect(
        color: .yellow,
        radius: 3
    )
    @State private var camera = Entity()

    var body: some View {
        RealityView { content in
            // Setup camera
            camera.components.set(PerspectiveCameraComponent())
            camera.position = [0, 0.5, 1.5]
            content.add(camera)

            // Load model
            if let model = try? await Entity(named: "gramophone") {
                model.generateCollisionShapes(recursive: true)
                model.components.set(InputTargetComponent(allowedInputTypes: .all))
                content.add(model)
            }

            // Configure effect
            let noRef: Entity? = nil
            effect.setViewMatrix(camera.transformMatrix(relativeTo: noRef).inverse)
            content.renderingEffects.customPostProcessing = .effect(effect)
        }
        .gesture(
            TapGesture()
                .targetedToAnyEntity()
                .onEnded { value in
                    effect.setSelection(value.entity)
                }
        )
    }
}
```

## API

### `PostProcessOutlineEffect`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `color` | `Color` | `.yellow` | The outline color |
| `radius` | `Int` | `3` | Outline thickness in pixels |

### Methods

- `setSelection(_ entity: Entity?)` — Call when selection changes. Extracts mesh geometry and queues it for GPU upload.
- `setViewMatrix(_ matrix: simd_float4x4)` — Call every frame to keep the mask projection aligned with the camera.

## Mesh Extraction

The effect extracts geometry from `MeshResource.contents`, which provides direct access to:

- Packed vertex positions
- Triangle indices
- Instance transforms

No buffer unwinding or topology reconstruction required. The implementation also includes a fallback path for procedurally generated meshes via `LowLevelMesh`.

## Performance Notes

- When nothing is selected, a simple blit copy runs instead of the three passes
- Textures are allocated fresh on every `postProcess` call — a production implementation would cache these
- Overhead when idle is effectively zero

## Why Not Other Techniques?

| Technique | Issues |
|-----------|--------|
| **Bounding Box** | Clips mesh or floats away; conveys location, not shape |
| **Inverted Hull** | Breaks on concavities, non-uniform scale, extreme model sizes |
| **Post-Process** | Consistent pixel width, handles all geometry correctly |

## Related

- [Blog Post: RealityKit Outline Selection](https://elkraneo.com/realitykit-outline-selection/)
- [PostProcessEffect | Apple Developer Documentation](https://developer.apple.com/documentation/realitykit/postprocesseffect)

## License

MIT License — See [LICENSE](LICENSE) for details.

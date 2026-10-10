![PostProcessOutline: pixel-exact selection outlines for RealityKit](.github/banner.png)

Pixel-exact selection outlines for RealityKit, drawn as a Metal post-process. Works with `ARView` (UIKit/AppKit, iOS 15+) and `RealityView` (SwiftUI, iOS 26+).

![Outline steps](https://elkraneo.com/content/images/2026/04/steps.gif)

How and why it works: [RealityKit Outline Selection](https://elkraneo.com/realitykit-outline-selection/).

## Features

- Constant pixel width, at any mesh complexity or distance
- Follows moving entities; outline several entities at once
- Never modifies your scene hierarchy or materials
- Color and radius can change live
- X-ray style: the selection stays visible behind other geometry

## Installation

Swift Package Manager:

```swift
.package(url: "https://github.com/elkraneo/PostProcessOutline.git", branch: "main")
```

## Usage

### ARView (UIKit / AppKit, iOS 15+ / macOS 12+)

```swift
import Combine
import PostProcessOutline

let outline = OutlineRenderer(cgColor: UIColor.systemYellow.cgColor, radius: 3)
var outlineInstallation: (any Cancellable)?

// After setting up the scene:
outlineInstallation = outline.install(on: arView)   // keep it; releasing it uninstalls
outline.preloadGeometry(of: [piece])                // optional: avoids a hitch on first select

// On tap:
outline.setSelection(arView.entity(at: point))      // nil clears
```

`install(on:)` sets `arView.renderCallbacks` (replacing existing ones) and updates the camera and entity transforms every frame. `entity(at:)` needs collision shapes (`generateCollisionShapes(recursive:)`).

### RealityView (SwiftUI, iOS 26+ / macOS 26+)

```swift
@State private var effect = PostProcessOutlineEffect(color: .yellow, radius: 3)
@State private var updates: EventSubscription?

RealityView { content in
    // ... add a camera entity and your models ...
    content.renderingEffects.customPostProcessing = .effect(effect)
    updates = content.subscribe(to: SceneEvents.Update.self) { [effect, camera] _ in
        effect.setViewMatrix(camera.transformMatrix(relativeTo: nil).inverse)
        effect.updateTransforms()
    }
}
.gesture(TapGesture().targetedToAnyEntity().onEnded { effect.setSelection($0.entity) })
```

## API

`OutlineRenderer` is the core. `PostProcessOutlineEffect` wraps one for `RealityView` and forwards the same calls.

| | |
|---|---|
| `setSelection(_:)` | One entity, several, or `nil`. Includes child meshes |
| `preloadGeometry(of:)` | Read meshes up front. First read is slow (~100 ms for 46k vertices), cached after that |
| `color`, `radius` | Live. Radius is clamped to `0...64` |
| `install(on:)` | `ARView` hookup. Returns a `Cancellable` |
| `setViewMatrix(_:)`, `updateTransforms()` | Per-frame updates (done for you by `install(on:)`) |
| `prepare(device:)`, `encode(...)` | Raw hooks to drive it from your own render callbacks |

## How it works

1. **Mask:** the selected meshes are rendered into an R8 silhouette texture.
2. **Dilate:** the silhouette grows by `radius` pixels and its interior is removed, leaving a ring.
3. **Composite:** the ring is blended over the frame in the outline color.

Nothing selected means a plain copy, so idle cost is close to zero.

## Requirements

- Xcode 26+, Swift 6
- iOS 15+ / macOS 12+ (`ARView`), iOS 26+ / macOS 26+ (`RealityView`)
- Not available on visionOS: neither `ARView` nor `PostProcessEffect` exist there

Run-tested on macOS 27. iOS and older OS versions are build-tested only.

## Example

`Examples/PostProcessOutlineExample.xcodeproj` shows both versions side by side. `ARViewOutlineViewController.swift` is a self-contained UIKit/AppKit sample you can copy.

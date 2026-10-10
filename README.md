![PostProcessOutline: pixel-exact selection outlines for RealityKit](.github/banner.png)

Pixel-exact selection outlines for RealityKit, as a Metal post-process. Constant pixel width, follows moving entities, supports multi-selection, and never touches your scene. [How it works →](https://elkraneo.com/realitykit-outline-selection/)

![Outline steps](https://elkraneo.com/content/images/2026/04/steps.gif)

## Install

```swift
.package(url: "https://github.com/elkraneo/PostProcessOutline.git", from: "1.0.0")
```

## ARView (UIKit / AppKit, iOS 15+)

```swift
let outline = OutlineRenderer(cgColor: UIColor.systemYellow.cgColor, radius: 3)
let installation = outline.install(on: arView)  // keep alive; replaces arView.renderCallbacks
outline.setSelection(arView.entity(at: point))  // nil clears
```

## RealityView (SwiftUI, iOS 26+)

```swift
@State private var effect = PostProcessOutlineEffect(color: .yellow, radius: 3)

content.renderingEffects.customPostProcessing = .effect(effect)
updates = content.subscribe(to: SceneEvents.Update.self) { _ in  // every frame
    effect.setViewMatrix(camera.transformMatrix(relativeTo: nil).inverse)
    effect.updateTransforms()
}
effect.setSelection(entity)
```

## Notes

- Call `preloadGeometry(of:)` after loading selectable entities. The first mesh read is slow (~100 ms for 46k vertices); selection is ~1 ms after that.
- `color` and `radius` (0…64) can change live.
- The outline is X-ray style: it stays visible behind other geometry.
- Not available on visionOS. Run-tested on macOS 27; iOS is build-tested.

See [`Examples/`](Examples) for a working app with both versions.

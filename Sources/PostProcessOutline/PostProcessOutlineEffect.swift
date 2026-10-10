//
//  PostProcessOutlineEffect.swift
//  PostProcessOutline
//
//  Created by Cristian Díaz on 29.03.26.
//

import Metal
import RealityKit
import SwiftUI
import simd

// MARK: - Effect

/// A `PostProcessEffect` that draws a pixel-exact (X-ray) selection outline
/// using an `OutlineRenderer`:
///
/// 1. **Mask** — renders the selected meshes into an R8 silhouette texture.
/// 2. **Dilate** — expands the mask by `radius` pixels to form the outline ring.
/// 3. **Composite** — blends the outline color over the original frame.
///
/// The effect is a thin value wrapper; copies share the same renderer, so
/// selection, color and radius changes apply to every copy.
///
/// Per frame, from a `SceneEvents.Update` subscription, call
/// `setViewMatrix(_:)` with the inverse camera transform and `updateTransforms()`.
@available(iOS 26.0, macOS 26.0, *)
@available(visionOS, unavailable)
public struct PostProcessOutlineEffect: PostProcessEffect {
	public let renderer: OutlineRenderer

	public init(renderer: OutlineRenderer) {
		self.renderer = renderer
	}

	public init(color: Color = .yellow, radius: Int = 3) {
		self.renderer = OutlineRenderer(color: Self.rgba(from: color), radius: radius)
	}

	// MARK: - Appearance

	/// Outline thickness in pixels. Forwards to the renderer.
	public var radius: Int {
		get { renderer.radius }
		nonmutating set { renderer.radius = newValue }
	}

	/// Sets the outline color. Takes effect on the next frame.
	public func setColor(_ color: Color) {
		renderer.color = Self.rgba(from: color)
	}

	// MARK: - Selection

	/// Call this (on @MainActor) whenever the selected entity changes.
	@MainActor
	public func setSelection(_ entity: Entity?) {
		renderer.setSelection(entity)
	}

	/// Call this (on @MainActor) whenever the selected entities change.
	@MainActor
	public func setSelection(_ entities: [Entity]) {
		renderer.setSelection(entities)
	}

	/// Reads and caches the geometry of `entities` so the first selection doesn't hitch.
	@MainActor
	public func preloadGeometry(of entities: [Entity]) {
		renderer.preloadGeometry(of: entities)
	}

	/// Call this every frame so moving entities stay outlined.
	@MainActor
	public func updateTransforms() {
		renderer.updateTransforms()
	}

	/// Call this every frame to keep the mask projection aligned with the camera.
	public func setViewMatrix(_ matrix: simd_float4x4) {
		renderer.setViewMatrix(matrix)
	}

	// MARK: - PostProcessEffect

	public func prepare(for device: any MTLDevice) {
		renderer.prepare(device: device)
	}

	public func postProcess(
		context: borrowing PostProcessEffectContext<any MTLCommandBuffer>
	) {
		renderer.encode(
			device: context.device,
			commandBuffer: context.commandBuffer,
			source: context.sourceColorTexture,
			target: context.targetColorTexture,
			projection: context.projection
		)
	}

	// MARK: - Helpers

	private static func rgba(from color: Color) -> SIMD4<Float> {
		let r = color.resolve(in: EnvironmentValues())
		return SIMD4(Float(r.red), Float(r.green), Float(r.blue), Float(r.opacity))
	}
}

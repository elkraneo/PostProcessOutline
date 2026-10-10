//
//  ARView+Outline.swift
//  PostProcessOutline
//
//  Created by Cristian Díaz on 29.03.26.
//

import Combine
import Foundation
import Metal
import ObjectiveC
import RealityKit

// MARK: - ARView adapter

@available(iOS 15.0, macOS 12.0, *)
@available(visionOS, unavailable)
extension OutlineRenderer {

	/// Draws the (X-ray) outline on top of `arView`'s frames.
	///
	/// Wires `ARView.renderCallbacks` (`prepareWithDevice` and `postProcess` —
	/// any existing callbacks are overwritten) and subscribes to
	/// `SceneEvents.Update` to call `setViewMatrix(_:)` with the inverse camera
	/// transform and `updateTransforms()` every frame.
	///
	/// Keep the returned `Cancellable` alive for as long as the outline should
	/// be drawn. Cancelling (or releasing) it removes the render callbacks and
	/// the scene subscription. Once a newer `install(on:)` call has taken over
	/// the same `ARView`, cancelling an older one leaves its callbacks alone.
	@MainActor
	public func install(on arView: ARView) -> any Cancellable {
		arView.renderCallbacks = Self.makeRenderCallbacks(for: self)

		let subscription = arView.scene.subscribe(to: SceneEvents.Update.self) {
			[weak self, weak arView] _ in
			guard let self, let arView else { return }
			self.setViewMatrix(arView.cameraTransform.matrix.inverse)
			self.updateTransforms()
		}

		let installation = OutlineARViewInstallation(arView: arView, subscription: subscription)
		OutlineARViewInstallation.setCurrent(installation, on: arView)
		return AnyCancellable(installation)
	}

	/// Builds the render callbacks outside any actor: RealityKit invokes them on
	/// its render thread, so they must not inherit @MainActor isolation.
	private nonisolated static func makeRenderCallbacks(
		for renderer: OutlineRenderer
	) -> ARView.RenderCallbacks {
		var callbacks = ARView.RenderCallbacks()
		callbacks.prepareWithDevice = { @Sendable device in
			renderer.prepare(device: device)
		}
		callbacks.postProcess = { @Sendable context in
			renderer.encode(
				device: context.device,
				commandBuffer: context.commandBuffer,
				source: context.sourceColorTexture,
				target: context.targetColorTexture,
				projection: context.projection
			)
		}
		return callbacks
	}
}

// MARK: - Installation token

/// Undoes `OutlineRenderer.install(on:)`. Cancelling is idempotent and may
/// happen on any thread; teardown always runs on the main actor.
@available(iOS 15.0, macOS 12.0, *)
@available(visionOS, unavailable)
private final class OutlineARViewInstallation: Cancellable, @unchecked Sendable {
	private let lock = NSLock()
	// Both cleared on the first cancel().
	private weak var arView: ARView?
	private var subscription: (any Cancellable)?

	init(arView: ARView, subscription: any Cancellable) {
		self.arView = arView
		self.subscription = subscription
	}

	func cancel() {
		if Thread.isMainThread {
			MainActor.assumeIsolated { teardown() }
		} else {
			DispatchQueue.main.async { [self] in
				MainActor.assumeIsolated { teardown() }
			}
		}
	}

	@MainActor
	private func teardown() {
		let (arView, subscription) = lock.withLock {
			defer {
				self.arView = nil
				self.subscription = nil
			}
			return (self.arView, self.subscription)
		}
		subscription?.cancel()
		// Only reset the callbacks if a later install(on:) hasn't replaced ours.
		if let arView, Self.current(on: arView) === self {
			arView.renderCallbacks = ARView.RenderCallbacks()
			Self.setCurrent(nil, on: arView)
		}
	}

	// MARK: Current installation per ARView

	private nonisolated(unsafe) static var currentKey: UInt8 = 0

	@MainActor
	static func setCurrent(_ installation: OutlineARViewInstallation?, on arView: ARView) {
		// Weak box, so the ARView doesn't keep the installation alive.
		let box = installation.map(WeakInstallation.init)
		objc_setAssociatedObject(arView, &currentKey, box, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
	}

	@MainActor
	private static func current(on arView: ARView) -> OutlineARViewInstallation? {
		(objc_getAssociatedObject(arView, &currentKey) as? WeakInstallation)?.installation
	}

	private final class WeakInstallation {
		weak var installation: OutlineARViewInstallation?
		init(_ installation: OutlineARViewInstallation) { self.installation = installation }
	}
}

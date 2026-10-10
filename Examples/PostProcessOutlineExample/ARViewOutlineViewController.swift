//
//  ARViewOutlineViewController.swift
//  PostProcessOutlineExample
//
//  Created by Cristian Díaz on 29.03.26.
//
//  ARView path (iOS 15+ / macOS 12+): the outline runs in `ARView.renderCallbacks`.
//  This file is self-contained: copy it into a UIKit (or AppKit) app that depends
//  on the PostProcessOutline package, and add a `gramophone.usdz` (or change the
//  model name below).
//

import Combine
import PostProcessOutline
import RealityKit

#if os(iOS)
import UIKit

/// Shows a model in a non-AR `ARView` and outlines whatever the user taps.
@available(iOS 15.0, *)
final class ARViewOutlineViewController: UIViewController {
	private var arView: ARView!

	/// Draws the outline. One renderer per view.
	private let renderer = OutlineRenderer(cgColor: UIColor.systemYellow.cgColor, radius: 3)

	/// Keeps the outline installed. Cancelling it (or releasing it) removes the
	/// render callbacks and the per-frame update subscription.
	private var outlineInstallation: (any Cancellable)?

	/// The turntable animation.
	private var sceneSubscriptions: [AnyCancellable] = []

	override func loadView() {
		// `.nonAR` renders a virtual scene through our own camera; no ARSession is needed.
		arView = ARView(frame: .zero, cameraMode: .nonAR, automaticallyConfigureSession: false)
		arView.environment.background = .color(.secondarySystemBackground)
		view = arView
	}

	override func viewDidLoad() {
		super.viewDidLoad()

		// 1. Install the outline: wires renderCallbacks and refreshes the camera and
		//    selected-entity transforms every frame.
		outlineInstallation = renderer.install(on: arView)

		// 2. Build the scene (camera, light, models) and preload the outline geometry.
		sceneSubscriptions = OutlineDemoScene.build(in: arView, renderer: renderer)

		// 3. Tap to select; tapping empty space clears the selection.
		let tap = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
		arView.addGestureRecognizer(tap)
	}

	@objc private func handleTap(_ recognizer: UITapGestureRecognizer) {
		let location = recognizer.location(in: arView)
		// `entity(at:)` hit-tests collision shapes, so the model needs `generateCollisionShapes`.
		renderer.setSelection(OutlineDemoScene.selectableRoot(of: arView.entity(at: location)))
	}
}

#elseif os(macOS)
import AppKit

/// AppKit twin of the UIKit controller above. Scene setup is shared via `OutlineDemoScene`.
@available(macOS 12.0, *)
final class ARViewOutlineViewController: NSViewController {
	private var arView: ARView!
	private let renderer = OutlineRenderer(cgColor: NSColor.systemYellow.cgColor, radius: 3)
	private var outlineInstallation: (any Cancellable)?
	private var sceneSubscriptions: [AnyCancellable] = []

	override func loadView() {
		// On macOS ARView always renders a virtual (non-AR) scene.
		arView = ARView(frame: NSRect(x: 0, y: 0, width: 800, height: 600))
		arView.environment.background = .color(.windowBackgroundColor)
		view = arView
	}

	override func viewDidLoad() {
		super.viewDidLoad()
		outlineInstallation = renderer.install(on: arView)
		sceneSubscriptions = OutlineDemoScene.build(in: arView, renderer: renderer)

		let click = NSClickGestureRecognizer(target: self, action: #selector(handleClick(_:)))
		arView.addGestureRecognizer(click)
	}

	@objc private func handleClick(_ recognizer: NSClickGestureRecognizer) {
		let location = recognizer.location(in: arView)
		renderer.setSelection(OutlineDemoScene.selectableRoot(of: arView.entity(at: location)))
	}
}
#endif

// MARK: - Shared scene setup

/// Builds the demo scene. Nothing here is outline-specific except
/// `generateCollisionShapes`, which `ARView.entity(at:)` needs for picking.
@available(iOS 15.0, macOS 12.0, *)
@MainActor
enum OutlineDemoScene {
	/// Name given to entities that should be selected as a whole.
	static let selectableName = "selectable"

	/// Adds a camera, a light, the gramophone and an orbiting box, and preloads
	/// their outline geometry so the first tap doesn't hitch.
	/// Returns subscriptions to keep alive.
	static func build(in arView: ARView, renderer: OutlineRenderer) -> [AnyCancellable] {
		let anchor = AnchorEntity(world: .zero)

		let camera = PerspectiveCamera()
		camera.position = [0, 0.5, 1.5]
		anchor.addChild(camera)

		let light = DirectionalLight()
		light.look(at: .zero, from: [1, 2, 2], relativeTo: nil)
		anchor.addChild(light)

		let box = ModelEntity(
			mesh: .generateBox(size: 0.15, cornerRadius: 0.02),
			materials: [SimpleMaterial(color: .systemBlue, isMetallic: false)]
		)
		box.name = selectableName
		box.generateCollisionShapes(recursive: false)
		anchor.addChild(box)
		renderer.preloadGeometry(of: [box])

		arView.scene.addAnchor(anchor)

		var gramophone: Entity?
		func addModel(_ model: Entity) {
			model.name = selectableName
			model.generateCollisionShapes(recursive: true)
			anchor.addChild(model)
			renderer.preloadGeometry(of: [model])
			gramophone = model
		}

		if #available(iOS 18.0, macOS 15.0, *) {
			// Load asynchronously so the view appears immediately.
			Task { @MainActor in
				do {
					addModel(try await Entity(named: "gramophone"))
				} catch {
					print("Failed to load gramophone: \(error)")
				}
			}
		} else {
			do {
				addModel(try Entity.load(named: "gramophone"))
			} catch {
				print("Failed to load gramophone: \(error)")
			}
		}

		var subscriptions: [AnyCancellable] = []

		// Spin the gramophone and orbit the box (same motion as the RealityView
		// demo) so the outline is shown following moving entities.
		var elapsed: TimeInterval = 0
		arView.scene.subscribe(to: SceneEvents.Update.self) { event in
			elapsed += event.deltaTime
			let t = Float(elapsed)
			gramophone?.orientation = simd_quatf(angle: t * 0.3, axis: [0, 1, 0])
			box.position = [cos(t * 0.8) * 0.6, 0.35, sin(t * 0.8) * 0.6]
			box.orientation = simd_quatf(angle: t, axis: normalize([1, 1, 0]))
		}
		.store(in: &subscriptions)

		return subscriptions
	}

	/// Taps hit individual collision parts; walk up to the whole model.
	/// Returns `nil` for empty space, which clears the selection.
	static func selectableRoot(of entity: Entity?) -> Entity? {
		var current = entity
		while let candidate = current {
			if candidate.name == selectableName { return candidate }
			current = candidate.parent
		}
		return entity
	}
}

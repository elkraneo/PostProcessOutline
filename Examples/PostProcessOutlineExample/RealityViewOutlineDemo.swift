//
//  RealityViewOutlineDemo.swift
//  PostProcessOutlineExample
//
//  Created by Cristian Díaz on 29.03.26.
//

import PostProcessOutline
import RealityKit
import SwiftUI

/// iOS 26 / macOS 26 path: `PostProcessOutlineEffect` installed as the
/// `RealityView`'s custom post-processing effect.
///
/// The gramophone spins and a box orbits around it, so the outline is shown
/// following moving entities (transforms are refreshed every frame).
@available(macOS 26.0, iOS 26.0, *)
struct RealityViewOutlineDemo: View {
	@State private var effect = PostProcessOutlineEffect(
		color: .yellow,
		radius: 3
	)
	@State private var camera = Entity()
	@State private var selectableEntities: [Entity] = []
	@State private var selectedEntity: Entity?
	/// Per-frame update subscription. Must be kept alive for the outline to track the camera and entities.
	@State private var updateSubscription: EventSubscription?

	var body: some View {
		RealityView { content in
			camera.components.set(PerspectiveCameraComponent())
			camera.position = [0, 0.5, 1.5]
			content.add(camera)

			var selectable: [Entity] = []

			let gramophone = try? await Entity(named: "gramophone")
			if let gramophone {
				gramophone.generateCollisionShapes(recursive: true)
				gramophone.components.set(InputTargetComponent(allowedInputTypes: .all))
				content.add(gramophone)
				selectable.append(gramophone)
			}

			let box = ModelEntity(
				mesh: .generateBox(size: 0.15, cornerRadius: 0.02),
				materials: [SimpleMaterial(color: .systemBlue, isMetallic: false)]
			)
			box.generateCollisionShapes(recursive: false)
			box.components.set(InputTargetComponent(allowedInputTypes: .all))
			content.add(box)
			selectable.append(box)

			selectableEntities = selectable
			// Read the outline geometry now so the first tap doesn't hitch.
			effect.preloadGeometry(of: selectable)
			content.renderingEffects.customPostProcessing = .effect(effect)

			// Animate first, then hand the outline the new camera and entity transforms.
			let effect = effect
			let camera = camera
			let noRef: Entity? = nil
			var elapsed: TimeInterval = 0
			updateSubscription = content.subscribe(to: SceneEvents.Update.self) { event in
				elapsed += event.deltaTime
				let t = Float(elapsed)

				gramophone?.orientation = simd_quatf(angle: t * 0.3, axis: [0, 1, 0])
				box.position = [cos(t * 0.8) * 0.6, 0.35, sin(t * 0.8) * 0.6]
				box.orientation = simd_quatf(angle: t, axis: normalize([1, 1, 0]))

				effect.setViewMatrix(camera.transformMatrix(relativeTo: noRef).inverse)
				effect.updateTransforms()
			}
		}
		.gesture(
			TapGesture()
				.targetedToAnyEntity()
				.onEnded { value in
					select(value.entity)
				}
		)
		.overlay(alignment: .bottom) {
			Button("Clear Selection") {
				select(nil)
			}
			.buttonStyle(.borderedProminent)
			.disabled(selectedEntity == nil)
			.padding()
		}
	}

	/// Selects the top-level model containing `entity` (taps hit the individual
	/// collision parts of the gramophone). Tapping the selected model again deselects it.
	private func select(_ entity: Entity?) {
		var root = entity
		while let candidate = root, !selectableEntities.contains(where: { $0 === candidate }) {
			root = candidate.parent
		}
		if let root, root === selectedEntity {
			selectedEntity = nil
		} else {
			selectedEntity = root
		}
		effect.setSelection(selectedEntity)
	}
}

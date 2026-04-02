//
//  ContentView.swift
//  PostProcessOutline
//
//  Created by Cristian Díaz on 29.03.26.
//

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
			camera.components.set(PerspectiveCameraComponent())
			camera.position = [0, 0.5, 1.5]
			content.add(camera)

			if let model = try? await Entity(named: "gramophone") {
				model.generateCollisionShapes(recursive: true)
				model.components.set(InputTargetComponent(allowedInputTypes: .all))
				content.add(model)
			}

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

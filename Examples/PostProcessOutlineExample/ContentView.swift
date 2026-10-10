//
//  ContentView.swift
//  PostProcessOutlineExample
//
//  Created by Cristian Díaz on 29.03.26.
//

import SwiftUI

/// Switches between the two ways of using PostProcessOutline:
/// the iOS 26 `RealityView` post-process effect and the iOS 15 `ARView` render callbacks.
struct ContentView: View {
	enum Demo: CaseIterable, Identifiable {
		case realityView
		case arView

		var id: Self { self }

		var title: String {
			switch self {
			case .realityView: "RealityView (iOS 26+)"
			case .arView: "ARView (iOS 15+)"
			}
		}
	}

	@State private var demo: Demo = .realityView

	var body: some View {
		VStack(spacing: 0) {
			Picker("Demo", selection: $demo) {
				ForEach(Demo.allCases) { demo in
					Text(demo.title).tag(demo)
				}
			}
			.pickerStyle(.segmented)
			.labelsHidden()
			.padding()

			switch demo {
			case .realityView:
				RealityViewOutlineDemo()
			case .arView:
				ARViewOutlineDemo()
			}
		}
	}
}

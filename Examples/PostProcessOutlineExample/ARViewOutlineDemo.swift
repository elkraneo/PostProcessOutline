//
//  ARViewOutlineDemo.swift
//  PostProcessOutlineExample
//
//  Created by Cristian Díaz on 29.03.26.
//

import SwiftUI

/// Hosts `ARViewOutlineViewController` in SwiftUI.
#if os(iOS)
struct ARViewOutlineDemo: UIViewControllerRepresentable {
	func makeUIViewController(context: Context) -> ARViewOutlineViewController {
		ARViewOutlineViewController()
	}

	func updateUIViewController(_ viewController: ARViewOutlineViewController, context: Context) {}
}
#elseif os(macOS)
struct ARViewOutlineDemo: NSViewControllerRepresentable {
	func makeNSViewController(context: Context) -> ARViewOutlineViewController {
		ARViewOutlineViewController()
	}

	func updateNSViewController(_ viewController: ARViewOutlineViewController, context: Context) {}
}
#endif

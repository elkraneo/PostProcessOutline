// swift-tools-version: 6.0

import PackageDescription

let package = Package(
	name: "PostProcessOutline",
	platforms: [
		.iOS(.v15),
		.macOS(.v12),
		// RealityKit's ARView / ModelComponent APIs exist on tvOS from 26.
		.tvOS("26.0"),
	],
	products: [
		.library(
			name: "PostProcessOutline",
			targets: ["PostProcessOutline"]
		)
	],
	targets: [
		.target(
			name: "PostProcessOutline",
			resources: [
				// Compiled into default.metallib inside the package resource bundle,
				// loaded at runtime via `Bundle.module`.
				.process("OutlineShaders.metal")
			]
		),
		.testTarget(
			name: "PostProcessOutlineTests",
			dependencies: ["PostProcessOutline"]
		),
	]
)

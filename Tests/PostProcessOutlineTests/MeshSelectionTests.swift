//
//  MeshSelectionTests.swift
//  PostProcessOutline
//

import RealityKit
import Testing

@testable import PostProcessOutline

#if !os(visionOS)

@MainActor
@Suite("Mesh selection")
struct MeshSelectionTests {

	/// parent (mesh) ── child (mesh)
	private func parentAndChild() -> (parent: ModelEntity, child: ModelEntity) {
		let parent = ModelEntity(mesh: .generateBox(size: 1), materials: [])
		let child = ModelEntity(mesh: .generateSphere(radius: 0.2), materials: [])
		parent.addChild(child)
		return (parent, child)
	}

	private func owners(
		_ entity: Entity,
		_ mode: OutlineRenderer.MeshSelection
	) -> [Entity?] {
		OutlineMeshExtraction.collect(from: entity, mode: mode, cache: OutlineGeometryCache())
			.map(\.owner)
	}

	@Test func subtreeIncludesOwnAndChildMeshes() {
		let (parent, child) = parentAndChild()
		let found = owners(parent, .subtree)
		#expect(found.contains { $0 === parent })
		#expect(found.contains { $0 === child })
	}

	@Test func nearestMeshesStopsAtOwnMesh() {
		let (parent, _) = parentAndChild()
		let found = owners(parent, .nearestMeshes)
		#expect(!found.isEmpty)
		#expect(found.allSatisfy { $0 === parent })
	}

	@Test func preferDescendantsSkipsAggregateParentMesh() {
		let (parent, child) = parentAndChild()
		let found = owners(parent, .preferDescendants)
		#expect(!found.isEmpty)
		#expect(found.allSatisfy { $0 === child })
	}

	@Test func preferDescendantsFallsBackToOwnMesh() {
		let leaf = ModelEntity(mesh: .generateBox(size: 1), materials: [])
		let found = owners(leaf, .preferDescendants)
		#expect(!found.isEmpty)
		#expect(found.allSatisfy { $0 === leaf })
	}

	@Test(arguments: [
		OutlineRenderer.MeshSelection.subtree, .nearestMeshes, .preferDescendants,
	])
	func groupWithoutMeshUsesAllChildren(mode: OutlineRenderer.MeshSelection) {
		let group = Entity()
		let a = ModelEntity(mesh: .generateBox(size: 1), materials: [])
		let b = ModelEntity(mesh: .generateBox(size: 1), materials: [])
		group.addChild(a)
		group.addChild(b)
		let found = owners(group, mode)
		#expect(found.contains { $0 === a })
		#expect(found.contains { $0 === b })
	}

	@Test func extractedGeometryIsTriangles() throws {
		let box = ModelEntity(mesh: .generateBox(size: 1), materials: [])
		let meshes = OutlineMeshExtraction.collect(
			from: box, mode: .subtree, cache: OutlineGeometryCache()
		)
		let mesh = try #require(meshes.first)
		#expect(!mesh.data.positions.isEmpty)
		#expect(mesh.data.indexCount > 0)
		#expect(mesh.data.indexCount % 3 == 0)
	}

	@Test func selectionBeforeActivationStillHasTransform() {
		// Not added to a scene, so not active yet.
		let box = ModelEntity(mesh: .generateBox(size: 1), materials: [])
		#expect(!box.isActive)
		let mesh = OutlineMeshExtraction.collect(
			from: box, mode: .subtree, cache: OutlineGeometryCache()
		).first
		#expect(mesh?.selectionModelMatrix != nil)
		#expect(mesh?.currentModelMatrix == nil)
	}

	@Test func cacheMissesInPlaceEditWithSameBoundsUnlessDisabled() throws {
		// A unit box and a radius-0.5 sphere share the same bounds.
		let entity = ModelEntity(mesh: .generateBox(size: 1), materials: [])
		let mesh = try #require(entity.model?.mesh)
		let cache = OutlineGeometryCache()
		let boxVertices = OutlineMeshExtraction.collect(from: entity, mode: .subtree, cache: cache)
			.first?.data.positions.count

		try mesh.replace(with: MeshResource.generateSphere(radius: 0.5).contents)
		let cached = OutlineMeshExtraction.collect(from: entity, mode: .subtree, cache: cache)
			.first?.data.positions.count
		let fresh = OutlineMeshExtraction.collect(from: entity, mode: .subtree, cache: OutlineGeometryCache())
			.first?.data.positions.count

		#expect(cached == boxVertices)  // documented limitation of the cache
		#expect(fresh != boxVertices)   // what `cachesGeometry = false` relies on
	}

	@Test func lowLevelMeshFloat4PositionsAreRead() throws {
		guard #available(macOS 15.0, iOS 18.0, tvOS 26.0, *) else { return }
		var descriptor = LowLevelMesh.Descriptor()
		descriptor.vertexAttributes = [.init(semantic: .position, format: .float4, offset: 0)]
		descriptor.vertexLayouts = [.init(bufferIndex: 0, bufferStride: 16)]
		descriptor.vertexCapacity = 3
		descriptor.indexCapacity = 3
		descriptor.indexType = .uint32
		let mesh = try LowLevelMesh(descriptor: descriptor)
		mesh.withUnsafeMutableBytes(bufferIndex: 0) { raw in
			let v = raw.bindMemory(to: SIMD4<Float>.self)
			v[0] = [0, 0, 0, 1]; v[1] = [1, 0, 0, 1]; v[2] = [0, 1, 0, 1]
		}
		mesh.withUnsafeMutableIndices { raw in
			let i = raw.bindMemory(to: UInt32.self)
			i[0] = 0; i[1] = 1; i[2] = 2
		}
		mesh.parts.replaceAll([
			LowLevelMesh.Part(indexCount: 3, topology: .triangle, bounds: BoundingBox(min: .zero, max: [1, 1, 0]))
		])
		let entity = ModelEntity(mesh: try MeshResource(from: mesh), materials: [])
		let found = OutlineMeshExtraction.collect(from: entity, mode: .subtree, cache: OutlineGeometryCache())
		let data = try #require(found.first?.data)
		#expect(data.indexCount == 3)
		#expect(data.positions.contains([1, 0, 0]))
	}

	@Test func colorConversionKeepsExtendedRange() {
		let c = OutlineRenderer.linear(fromSRGB: [1.2, -0.1, 0.5, 0.8])
		#expect(c.x > 1)                              // wide-gamut component not clipped
		#expect(c.y < 0)
		#expect(abs(c.z - 0.2140) < 0.001)           // standard sRGB 0.5 → linear
		#expect(c.w == 0.8)                           // alpha unchanged
	}

	@Test func rendererClampsRadius() {
		let renderer = OutlineRenderer(radius: 10_000)
		#expect(renderer.radius == OutlineRenderer.maxRadius)
		renderer.radius = -5
		#expect(renderer.radius == 0)
	}
}

#endif

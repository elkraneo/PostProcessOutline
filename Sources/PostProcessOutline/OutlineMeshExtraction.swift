//
//  OutlineMeshExtraction.swift
//  PostProcessOutline
//
//  Created by Cristian Díaz on 29.03.26.
//

import Metal
import RealityKit
import simd

// MARK: - Mesh data
//
// Selection geometry is read once on @MainActor when the selection changes,
// packed into plain arrays, and handed to the render thread for GPU upload.
// Per-frame movement is handled by refreshing model matrices only.

/// CPU-side geometry for one mesh part, ready for GPU upload.
struct OutlineMeshData: Sendable {
	let positions: [SIMD3<Float>]  // Metal float3, stride 16 — uploaded as is
	let indices: [UInt8]
	let indexCount: Int
	let indexType: MTLIndexType
}

/// One selected mesh: its geometry, the entity that owns it, and the
/// transform from that entity's space into mesh space (e.g. a
/// `MeshResource.Instance` transform).
///
/// The owner is held weakly so a selection never keeps an entity alive.
/// The model matrix is `owner world transform × localTransform`, re-read
/// every frame via `OutlineRenderer.updateTransforms()`.
@available(iOS 15.0, macOS 12.0, tvOS 26.0, *)
@available(visionOS, unavailable)
struct OutlineSelectedMesh: Sendable {
	weak var owner: Entity?
	let localTransform: simd_float4x4
	let data: OutlineMeshData

	/// Current model matrix, or `nil` when the owner is gone, disabled or no
	/// longer anchored in a scene (in which case the mesh is skipped this frame).
	/// Used by the per-frame `updateTransforms()`.
	@MainActor
	var currentModelMatrix: simd_float4x4? {
		guard let owner, owner.isActive else { return nil }
		return modelMatrix(of: owner)
	}

	/// Model matrix taken at selection time. Doesn't require the owner to be
	/// active yet: an entity is often selected before it's anchored, and
	/// callers that never call `updateTransforms()` must still see it.
	@MainActor
	var selectionModelMatrix: simd_float4x4? {
		guard let owner else { return nil }
		return modelMatrix(of: owner)
	}

	@MainActor
	private func modelMatrix(of owner: Entity) -> simd_float4x4 {
		let noRef: Entity? = nil
		return simd_mul(owner.transformMatrix(relativeTo: noRef), localTransform)
	}
}

// MARK: - Mesh collection (@MainActor)

@available(iOS 15.0, macOS 12.0, tvOS 26.0, *)
@available(visionOS, unavailable)
@MainActor
enum OutlineMeshExtraction {

	/// Walks the entity subtree and packs position/index data for the meshes
	/// that `mode` selects.
	static func collect(
		from entity: Entity,
		mode: OutlineRenderer.MeshSelection,
		cache: OutlineGeometryCache
	) -> [OutlineSelectedMesh] {
		cache.pruneReleasedEntities()
		var meshes: [OutlineSelectedMesh] = []
		collect(from: entity, mode: mode, cache: cache, into: &meshes)
		return meshes
	}

	private static func collect(
		from entity: Entity,
		mode: OutlineRenderer.MeshSelection,
		cache: OutlineGeometryCache,
		into meshes: inout [OutlineSelectedMesh]
	) {
		switch mode {
		case .subtree:
			extract(from: entity, cache: cache, into: &meshes)
			collectChildren(of: entity, mode: mode, cache: cache, into: &meshes)

		case .nearestMeshes:
			if !extract(from: entity, cache: cache, into: &meshes) {
				collectChildren(of: entity, mode: mode, cache: cache, into: &meshes)
			}

		case .preferDescendants:
			let start = meshes.count
			collectChildren(of: entity, mode: mode, cache: cache, into: &meshes)
			if meshes.count == start {
				extract(from: entity, cache: cache, into: &meshes)
			}
		}
	}

	private static func collectChildren(
		of entity: Entity,
		mode: OutlineRenderer.MeshSelection,
		cache: OutlineGeometryCache,
		into meshes: inout [OutlineSelectedMesh]
	) {
		for child in entity.children {
			collect(from: child, mode: mode, cache: cache, into: &meshes)
		}
	}

	/// Appends `entity`'s own mesh, if it has one. Returns whether it did.
	@discardableResult
	private static func extract(
		from entity: Entity,
		cache: OutlineGeometryCache,
		into meshes: inout [OutlineSelectedMesh]
	) -> Bool {
		guard let model = entity.components[ModelComponent.self] else { return false }

		// LowLevelMesh first, when the mesh has one: its vertex descriptor gives
		// the real stride. For LowLevelMesh-backed meshes (USDKit's
		// USDStageComponent), MeshResource.Contents reads the interleaved vertex
		// buffer as packed positions — on macOS 27 a 13,482-vertex mesh came back
		// as 89,880 "positions" starting with a normal — which draws spikes.
		if #available(iOS 18.0, macOS 15.0, *),
			let low = model.mesh.lowLevelMesh,
			let data = extractFromLowLevel(low)
		{
			meshes.append(
				OutlineSelectedMesh(owner: entity, localTransform: matrix_identity_float4x4, data: data)
			)
			return true
		}

		// MeshResource.Contents (most USD imports and generated meshes).
		let fromContents = cache.contents(of: model.mesh, owner: entity) {
			extractFromContents($0.contents)
		}
		if !fromContents.isEmpty {
			meshes.append(
				contentsOf: fromContents.map {
					OutlineSelectedMesh(owner: entity, localTransform: $0.transform, data: $0.data)
				}
			)
			return true
		}
		return false
	}

	private static func extractFromContents(
		_ contents: MeshResource.Contents
	) -> [(data: OutlineMeshData, transform: simd_float4x4)] {
		let modelsByID = Dictionary(
			contents.models.map { ($0.id, $0) },
			uniquingKeysWith: { first, _ in first }
		)
		let instances = Array(contents.instances)

		// Instance transforms are relative to the owning entity; the entity's
		// world transform is applied per frame by the renderer.
		let pairs: [(MeshResource.Model, simd_float4x4)] =
			instances.isEmpty
			? contents.models.map { ($0, matrix_identity_float4x4) }
			: instances.compactMap { instance in
				guard let model = modelsByID[instance.model] else { return nil }
				return (model, instance.transform)
			}

		return pairs.flatMap { model, transform in
			model.parts.compactMap { part -> (data: OutlineMeshData, transform: simd_float4x4)? in
				guard let indices = part.triangleIndices?.elements, !indices.isEmpty
				else { return nil }
				// Same layout as the shader's float3, so no per-vertex repacking.
				let positions = part.positions.elements
				guard !positions.isEmpty else { return nil }
				let data = OutlineMeshData(
					positions: positions,
					indices: indices.withUnsafeBytes { Array($0) },
					indexCount: indices.count,
					indexType: .uint32
				)
				return (data, transform)
			}
		}
	}

	@available(iOS 18.0, macOS 15.0, tvOS 26.0, *)
	private static func extractFromLowLevel(_ mesh: LowLevelMesh) -> OutlineMeshData? {
		let descriptor = mesh.descriptor
		guard
			let posAttr = descriptor.vertexAttributes.first(where: {
				$0.semantic == .position
			}),
			descriptor.vertexLayouts.indices.contains(posAttr.layoutIndex)
		else { return nil }

		// float3 and float4 positions are supported (XYZ is read into SIMD3 for
		// the mask vertex shader); other formats are skipped.
		guard posAttr.format == .float3 || posAttr.format == .float4 else { return nil }

		let layout = descriptor.vertexLayouts[posAttr.layoutIndex]
		let stride = layout.bufferStride
		let posOffset = layout.bufferOffset + posAttr.offset
		guard stride >= 12 else { return nil }

		var positions: [SIMD3<Float>] = []
		mesh.withUnsafeBytes(bufferIndex: layout.bufferIndex) { raw in
			guard raw.count >= posOffset + 12 else { return }
			let count = min(
				mesh.vertexCapacity,
				(raw.count - posOffset - 12) / stride + 1
			)
			positions.reserveCapacity(count)
			for i in 0..<count {
				let base = i * stride + posOffset
				positions.append(SIMD3(
					raw.loadUnaligned(fromByteOffset: base, as: Float.self),
					raw.loadUnaligned(fromByteOffset: base + 4, as: Float.self),
					raw.loadUnaligned(fromByteOffset: base + 8, as: Float.self)
				))
			}
		}

		let indexType = descriptor.indexType
		let indexSize: Int
		switch indexType {
		case .uint32: indexSize = 4
		case .uint16: indexSize = 2
		@unknown default: return nil
		}

		// Gather the index ranges of all triangle parts.
		var indexBytes: [UInt8] = []
		mesh.withUnsafeIndices { raw in
			for part in mesh.parts where part.topology == .triangle {
				// `indexOffset` is already in bytes.
				let start = part.indexOffset
				let end = start + part.indexCount * indexSize
				guard part.indexCount > 0, start >= 0, end <= raw.count else { continue }
				indexBytes.append(contentsOf: raw[start..<end])
			}
		}
		let indexCount = indexBytes.count / indexSize

		guard !positions.isEmpty, indexCount > 0 else { return nil }
		return OutlineMeshData(
			positions: positions,
			indices: indexBytes,
			indexCount: indexCount,
			indexType: indexType
		)
	}
}

// MARK: - Geometry cache

/// Extracted `MeshResource.Contents` geometry per entity, owned by one
/// `OutlineRenderer` so it's released together with the renderer.
///
/// Reading MeshBuffer elements is slow (~100 ms for a 46k-vertex model, in
/// Release too), so each entity's mesh is read once and reused on later
/// selections. LowLevelMesh isn't cached: it's read with a raw copy, and its
/// buffers are commonly updated in place.
///
/// Keyed by entity, not MeshResource: for imported models RealityKit hands out
/// a temporary MeshResource wrapper per access, so its identity isn't stable.
/// Holding the wrapper keeps it stable while the entry lives, which lets `===`
/// detect a swapped mesh; a bounds check catches most in-place
/// `replace(with:)` edits.
///
/// Only touched on @MainActor (through `OutlineMeshExtraction`).
@available(iOS 15.0, macOS 12.0, tvOS 26.0, *)
@available(visionOS, unavailable)
final class OutlineGeometryCache: @unchecked Sendable {
	typealias Geometry = [(data: OutlineMeshData, transform: simd_float4x4)]

	private struct Entry {
		weak var entity: Entity?
		let mesh: MeshResource
		let bounds: BoundingBox
		let geometry: Geometry
	}

	private var entries: [ObjectIdentifier: Entry] = [:]

	@MainActor
	func contents(
		of mesh: MeshResource,
		owner entity: Entity,
		extract: (MeshResource) -> Geometry
	) -> Geometry {
		let key = ObjectIdentifier(entity)
		let bounds = mesh.bounds
		// The weak reference guards against a reused ObjectIdentifier.
		if let entry = entries[key], entry.entity === entity,
			entry.mesh === mesh, entry.bounds == bounds
		{
			return entry.geometry
		}
		let geometry = extract(mesh)
		entries[key] = Entry(entity: entity, mesh: mesh, bounds: bounds, geometry: geometry)
		return geometry
	}

	/// Drops entries whose entity has been released.
	@MainActor
	func pruneReleasedEntities() {
		entries = entries.filter { $0.value.entity != nil }
	}
}

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

// MARK: - Thread-safe render state
//
// PostProcessEffect callbacks are nonisolated, but mesh data is collected on
// @MainActor. OutlineRenderState bridges the two with a lock.

@available(macOS 26.0, iOS 26.0, *)
@available(visionOS, unavailable)
final class OutlineRenderState: @unchecked Sendable {

	// Pending mesh data written on @MainActor, consumed inside postProcess.
	struct PendingMesh {
		let positions: [Float]  // packed xyz, stride 12
		let indices: [UInt8]
		let indexCount: Int
		let indexType: MTLIndexType
		let modelMatrix: simd_float4x4
	}

	// GPU-resident mesh data, only accessed from postProcess.
	struct GPUMesh {
		let vertexBuffer: any MTLBuffer
		let indexBuffer: any MTLBuffer
		let indexCount: Int
		let indexType: MTLIndexType
		let modelMatrix: simd_float4x4
	}

	private let lock = NSLock()
	private var _pending: [PendingMesh]? = nil

	var pending: [PendingMesh]? {
		get { lock.withLock { _pending } }
		set { lock.withLock { _pending = newValue } }
	}

	// Set from @MainActor update closure, read inside postProcess.
	var viewMatrix: simd_float4x4 = matrix_identity_float4x4

	// Compiled once in prepare(for:).
	var maskPipeline: (any MTLRenderPipelineState)?
	var dilatePipeline: (any MTLComputePipelineState)?
	var compositePipeline: (any MTLComputePipelineState)?

	var gpuMeshes: [GPUMesh] = []

	// MARK: Pipeline compilation

	func buildPipelines(device: any MTLDevice) {
		guard let library = device.makeDefaultLibrary() else { return }

		let maskDesc = MTLRenderPipelineDescriptor()
		maskDesc.vertexFunction = library.makeFunction(name: "outlineMaskVertex")
		maskDesc.fragmentFunction = library.makeFunction(
			name: "outlineMaskFragment"
		)
		maskDesc.colorAttachments[0].pixelFormat = .r8Unorm
		maskDesc.depthAttachmentPixelFormat = .depth32Float
		let vd = MTLVertexDescriptor()
		vd.attributes[0].format = .float3
		vd.attributes[0].offset = 0
		vd.attributes[0].bufferIndex = 0
		vd.layouts[0].stride = 12
		maskDesc.vertexDescriptor = vd
		maskPipeline = try? device.makeRenderPipelineState(descriptor: maskDesc)

		if let dilate = library.makeFunction(name: "outlineDilate") {
			dilatePipeline = try? device.makeComputePipelineState(function: dilate)
		}
		if let composite = library.makeFunction(name: "outlineComposite") {
			compositePipeline = try? device.makeComputePipelineState(
				function: composite
			)
		}
	}

	// MARK: CPU → GPU upload (called from postProcess)

	func uploadPending(device: any MTLDevice) {
		guard let entries = pending else { return }
		pending = nil
		gpuMeshes = entries.compactMap { entry in
			guard
				let vb = device.makeBuffer(
					bytes: entry.positions,
					length: entry.positions.count * MemoryLayout<Float>.size,
					options: .storageModeShared
				),
				let ib = device.makeBuffer(
					bytes: entry.indices,
					length: entry.indices.count,
					options: .storageModeShared
				)
			else { return nil }
			return GPUMesh(
				vertexBuffer: vb,
				indexBuffer: ib,
				indexCount: entry.indexCount,
				indexType: entry.indexType,
				modelMatrix: entry.modelMatrix
			)
		}
	}
}

// MARK: - Effect

/// A `PostProcessEffect` that draws a pixel-exact selection outline in three passes:
///
/// 1. **Mask** — renders the selected mesh into an R8 silhouette texture.
/// 2. **Dilate** — expands the mask by `radius` pixels to form the outline ring.
/// 3. **Composite** — blends the outline color over the original frame.
@available(macOS 26.0, iOS 26.0, *)
@available(visionOS, unavailable)
public struct PostProcessOutlineEffect: PostProcessEffect {
	public var color: Color
	public var radius: Int

	private let state = OutlineRenderState()

	public init(color: Color = .yellow, radius: Int = 3) {
		self.color = color
		self.radius = radius
	}

	// MARK: - Selection

	/// Call this (on @MainActor) whenever the selected entity changes.
	/// Walks the entity subtree, packs position data, and queues it for GPU upload.
	@MainActor
	public func setSelection(_ entity: Entity?) {
		guard let entity else {
			state.pending = []
			return
		}
		var meshes: [OutlineRenderState.PendingMesh] = []
		collect(from: entity, into: &meshes)
		state.pending = meshes
	}

	/// Call this every frame from the RealityView update closure to keep the
	/// mask projection aligned with the camera.
	public func setViewMatrix(_ matrix: simd_float4x4) {
		state.viewMatrix = matrix
	}

	// MARK: - Mesh collection (@MainActor)

	@MainActor
	private func collect(
		from entity: Entity,
		into meshes: inout [OutlineRenderState.PendingMesh]
	) {
		// If this entity owns geometry, outline just that mesh — don't descend
		// into children (avoids outlining the whole imported hierarchy).
		if extract(from: entity, into: &meshes) { return }
		for child in entity.children {
			collect(from: child, into: &meshes)
		}
	}

	@MainActor
	private func extract(
		from entity: Entity,
		into meshes: inout [OutlineRenderState.PendingMesh]
	) -> Bool {
		guard let model = entity.components[ModelComponent.self] else {
			return false
		}
		let noRef: Entity? = nil
		let worldTransform = entity.transformMatrix(relativeTo: noRef)

		// Primary path: MeshResource.Contents (works for most USD imports).
		let fromContents = extractFromContents(
			model.mesh.contents,
			worldTransform: worldTransform
		)
		if !fromContents.isEmpty {
			meshes.append(contentsOf: fromContents)
			return true
		}

		// Fallback: LowLevelMesh (needed for procedurally generated meshes).
		if let low = model.mesh.lowLevelMesh,
			let mesh = extractFromLowLevel(low, worldTransform: worldTransform)
		{
			meshes.append(mesh)
			return true
		}

		return false
	}

	@MainActor
	private func extractFromContents(
		_ contents: MeshResource.Contents,
		worldTransform: simd_float4x4
	) -> [OutlineRenderState.PendingMesh] {
		let modelsByID = Dictionary(
			uniqueKeysWithValues: contents.models.map { ($0.id, $0) }
		)
		let instances = Array(contents.instances)

		let pairs: [(MeshResource.Model, simd_float4x4)] =
			instances.isEmpty
			? contents.models.map { ($0, worldTransform) }
			: instances.compactMap { instance in
				guard let model = modelsByID[instance.model] else { return nil }
				return (model, simd_mul(worldTransform, instance.transform))
			}

		return pairs.flatMap { model, transform in
			model.parts.compactMap { part -> OutlineRenderState.PendingMesh? in
				guard let indices = part.triangleIndices?.elements, !indices.isEmpty
				else { return nil }
				let positions = part.positions.elements.flatMap { [$0.x, $0.y, $0.z] }
				guard !positions.isEmpty else { return nil }
				return .init(
					positions: positions,
					indices: indices.withUnsafeBytes { Array($0) },
					indexCount: indices.count,
					indexType: .uint32,
					modelMatrix: transform
				)
			}
		}
	}

	@MainActor
	private func extractFromLowLevel(
		_ mesh: LowLevelMesh,
		worldTransform: simd_float4x4
	) -> OutlineRenderState.PendingMesh? {
		let descriptor = mesh.descriptor
		guard
			let posAttr = descriptor.vertexAttributes.first(where: {
				$0.semantic == .position
			})
		else { return nil }

		let layoutIndex = posAttr.layoutIndex
		let stride = descriptor.vertexLayouts[layoutIndex].bufferStride
		let posOffset = posAttr.offset

		var positions: [Float] = []
		mesh.withUnsafeBytes(bufferIndex: layoutIndex) { raw in
			let bytes = raw.bindMemory(to: UInt8.self)
			let count = raw.count / stride
			positions.reserveCapacity(count * 3)
			for i in 0..<count {
				let base = i * stride + posOffset
				var x: Float = 0
				var y: Float = 0
				var z: Float = 0
				withUnsafeMutableBytes(of: &x) {
					$0.copyBytes(from: bytes[base..<base + 4])
				}
				withUnsafeMutableBytes(of: &y) {
					$0.copyBytes(from: bytes[(base + 4)..<(base + 8)])
				}
				withUnsafeMutableBytes(of: &z) {
					$0.copyBytes(from: bytes[(base + 8)..<(base + 12)])
				}
				positions += [x, y, z]
			}
		}

		var indexBytes: [UInt8] = []
		mesh.withUnsafeIndices { indexBytes = Array($0) }

		let indexType = descriptor.indexType
		let indexCount: Int
		switch indexType {
		case .uint32: indexCount = indexBytes.count / 4
		case .uint16: indexCount = indexBytes.count / 2
		@unknown default: return nil
		}

		guard !positions.isEmpty, indexCount > 0 else { return nil }
		return .init(
			positions: positions,
			indices: indexBytes,
			indexCount: indexCount,
			indexType: indexType,
			modelMatrix: worldTransform
		)
	}

	// MARK: - PostProcessEffect

	public mutating func prepare(for device: any MTLDevice) {
		state.buildPipelines(device: device)
	}

	public mutating func postProcess(
		context: borrowing PostProcessEffectContext<any MTLCommandBuffer>
	) {
		let device = context.device
		state.uploadPending(device: device)

		guard
			!state.gpuMeshes.isEmpty,
			let maskPipeline = state.maskPipeline,
			let dilatePipeline = state.dilatePipeline,
			let compositePipeline = state.compositePipeline
		else {
			blit(
				from: context.sourceColorTexture,
				to: context.targetColorTexture,
				cb: context.commandBuffer
			)
			return
		}

		let w = context.sourceColorTexture.width
		let h = context.sourceColorTexture.height
		let cb = context.commandBuffer

		guard
			let maskTex = makeTexture(
				device: device,
				w: w,
				h: h,
				format: .r8Unorm,
				usage: [.renderTarget, .shaderRead]
			),
			let depthTex = makeTexture(
				device: device,
				w: w,
				h: h,
				format: .depth32Float,
				usage: .renderTarget
			),
			let edgeTex = makeTexture(
				device: device,
				w: w,
				h: h,
				format: .r8Unorm,
				usage: [.shaderRead, .shaderWrite]
			)
		else { return }

		// Pass 1 — render mesh silhouettes into the mask texture.
		let rpd = MTLRenderPassDescriptor()
		rpd.colorAttachments[0].texture = maskTex
		rpd.colorAttachments[0].loadAction = .clear
		rpd.colorAttachments[0].storeAction = .store
		rpd.depthAttachment.texture = depthTex
		rpd.depthAttachment.loadAction = .clear
		rpd.depthAttachment.storeAction = .dontCare
		rpd.depthAttachment.clearDepth = 1.0
		guard let renc = cb.makeRenderCommandEncoder(descriptor: rpd) else {
			return
		}
		renc.setRenderPipelineState(maskPipeline)
		for mesh in state.gpuMeshes {
			var mvp = context.projection * state.viewMatrix * mesh.modelMatrix
			renc.setVertexBuffer(mesh.vertexBuffer, offset: 0, index: 0)
			renc.setVertexBytes(
				&mvp,
				length: MemoryLayout<simd_float4x4>.size,
				index: 1
			)
			renc.drawIndexedPrimitives(
				type: .triangle,
				indexCount: mesh.indexCount,
				indexType: mesh.indexType,
				indexBuffer: mesh.indexBuffer,
				indexBufferOffset: 0
			)
		}
		renc.endEncoding()

		// Pass 2 — dilate the mask into an outline ring.
		guard let cenc1 = cb.makeComputeCommandEncoder() else { return }
		cenc1.setComputePipelineState(dilatePipeline)
		cenc1.setTexture(maskTex, index: 0)
		cenc1.setTexture(edgeTex, index: 1)
		var r = Int32(radius)
		cenc1.setBytes(&r, length: MemoryLayout<Int32>.size, index: 0)
		dispatch(cenc1, pipeline: dilatePipeline, w: w, h: h)
		cenc1.endEncoding()

		// Pass 3 — composite the outline color over the source frame.
		guard let cenc2 = cb.makeComputeCommandEncoder() else { return }
		cenc2.setComputePipelineState(compositePipeline)
		cenc2.setTexture(context.sourceColorTexture, index: 0)
		cenc2.setTexture(edgeTex, index: 1)
		cenc2.setTexture(context.targetColorTexture, index: 2)
		var c = resolvedColor
		cenc2.setBytes(&c, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
		dispatch(cenc2, pipeline: compositePipeline, w: w, h: h)
		cenc2.endEncoding()
	}

	// MARK: - Helpers

	private var resolvedColor: SIMD4<Float> {
		let r = color.resolve(in: EnvironmentValues())
		return SIMD4(Float(r.red), Float(r.green), Float(r.blue), Float(r.opacity))
	}

	private func dispatch(
		_ encoder: any MTLComputeCommandEncoder,
		pipeline: any MTLComputePipelineState,
		w: Int,
		h: Int
	) {
		let tg = MTLSize(width: 16, height: 16, depth: 1)
		let grid = MTLSize(width: (w + 15) / 16, height: (h + 15) / 16, depth: 1)
		encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
	}

	private func makeTexture(
		device: any MTLDevice,
		w: Int,
		h: Int,
		format: MTLPixelFormat,
		usage: MTLTextureUsage
	) -> (any MTLTexture)? {
		let desc = MTLTextureDescriptor.texture2DDescriptor(
			pixelFormat: format,
			width: w,
			height: h,
			mipmapped: false
		)
		desc.usage = usage
		desc.storageMode = .private
		return device.makeTexture(descriptor: desc)
	}

	private func blit(
		from source: any MTLTexture,
		to target: any MTLTexture,
		cb: any MTLCommandBuffer
	) {
		guard let enc = cb.makeBlitCommandEncoder() else { return }
		enc.copy(
			from: source,
			sourceSlice: 0,
			sourceLevel: 0,
			sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
			sourceSize: MTLSize(width: source.width, height: source.height, depth: 1),
			to: target,
			destinationSlice: 0,
			destinationLevel: 0,
			destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
		)
		enc.endEncoding()
	}
}

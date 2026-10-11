//
//  OutlineRenderer.swift
//  PostProcessOutline
//
//  Created by Cristian Díaz on 29.03.26.
//

import CoreGraphics
import Foundation
import Metal
import RealityKit
import os
import simd

// MARK: - Renderer

/// Draws a pixel-exact selection outline over a rendered RealityKit frame in three passes:
///
/// 1. **Mask** — renders the selected meshes into an R8 silhouette texture.
/// 2. **Dilate** — expands the mask by `radius` pixels to form the outline ring.
/// 3. **Composite** — blends the outline color over the original frame.
///
/// The outline is an **X-ray** outline: the mask pass doesn't test against the
/// scene's depth buffer, so the outline of a selected entity stays visible
/// through any geometry that occludes it.
///
/// `OutlineRenderer` is renderer-agnostic. Hook it up with
/// `install(on:)` for `ARView`, or wrap it in `PostProcessOutlineEffect`
/// for `RealityView` (iOS 26 / macOS 26).
///
/// One renderer can serve several views: intermediate textures are kept per
/// command queue.
///
/// ## Threading
/// Selection and transforms are read on `@MainActor`; `prepare(device:)` and
/// `encode(...)` run on RealityKit's render thread. All state shared between
/// the two is protected by a lock, so `color`, `radius` and
/// `setViewMatrix(_:)` may be called from any thread.
@available(iOS 15.0, macOS 12.0, tvOS 26.0, *)
@available(visionOS, unavailable)
public final class OutlineRenderer: @unchecked Sendable {

	// MARK: Shared state (lock-protected)

	/// State written from @MainActor / any thread, consumed by the render thread.
	private struct SharedState {
		// Selected meshes (owners held weakly), only touched on @MainActor.
		var selection: [OutlineSelectedMesh] = []
		// Geometry of a new selection waiting for upload, consumed in encode.
		var pendingUpload: [OutlineMeshData]? = nil
		// One entry per selected mesh; nil = skip this frame (owner gone / disabled).
		var modelMatrices: [simd_float4x4?] = []
		var viewMatrix: simd_float4x4 = matrix_identity_float4x4
		var color: SIMD4<Float>
		var radius: Int
		var meshSelection: MeshSelection = .subtree
		var cachesGeometry = true
	}

	private let lock = NSLock()
	private var shared: SharedState

	// Extracted mesh geometry, only touched on @MainActor.
	private let geometryCache = OutlineGeometryCache()

	// MARK: Render state (render-lock-protected)

	// GPU-resident mesh data, index-aligned with `SharedState.modelMatrices`.
	private struct GPUMesh {
		let vertexBuffer: any MTLBuffer
		let indexBuffer: any MTLBuffer
		let indexCount: Int
		let indexType: MTLIndexType
	}

	/// Resources owned by the render thread.
	private struct RenderState {
		var device: (any MTLDevice)?

		// Compiled once per device in prepare(device:).
		var maskPipeline: (any MTLRenderPipelineState)?
		var dilatePipeline: (any MTLComputePipelineState)?
		var compositePipeline: (any MTLComputePipelineState)?

		// nil entries are meshes whose buffers couldn't be allocated.
		var gpuMeshes: [GPUMesh?] = []

		// Intermediate textures per command queue, recreated only when the
		// frame size changes. Per queue, because frames on different queues
		// (e.g. one effect shared by two views) can run on the GPU at once.
		var intermediates: [ObjectIdentifier: IntermediateTextures] = [:]
	}

	private struct IntermediateTextures {
		weak var queue: (any MTLCommandQueue)?
		let mask: any MTLTexture
		let depth: any MTLTexture
		let edge: any MTLTexture
	}

	private let renderLock = NSLock()
	private var render = RenderState()

	private static let logger = Logger(
		subsystem: "PostProcessOutline",
		category: "OutlineRenderer"
	)

	// MARK: - Init

	/// Creates a renderer.
	/// - Parameters:
	///   - color: Outline color as sRGB-encoded RGBA in 0…1 (the values a color
	///     picker shows). Alpha controls the blend over the frame.
	///   - radius: Outline thickness in pixels, clamped to `0...maxRadius`.
	public init(color: SIMD4<Float> = [1, 1, 0, 1], radius: Int = 3) {
		self.shared = SharedState(color: color, radius: Self.clampRadius(radius))
	}

	/// Creates a renderer from a `CGColor`, converted to sRGB components.
	public convenience init(cgColor: CGColor, radius: Int = 3) {
		self.init(color: Self.rgba(from: cgColor), radius: radius)
	}

	// MARK: - Appearance

	/// Outline color as sRGB-encoded RGBA in 0…1. Takes effect on the next frame.
	/// It's converted to linear for blending, since RealityKit's frame is linear.
	public var color: SIMD4<Float> {
		get { lock.withLock { shared.color } }
		set { lock.withLock { shared.color = newValue } }
	}

	/// Largest supported outline radius. The dilate pass scans a (2r+1)²
	/// neighborhood per pixel, so larger values would stall the GPU.
	public static let maxRadius = 64

	private static func clampRadius(_ radius: Int) -> Int {
		min(max(radius, 0), maxRadius)
	}

	/// Outline thickness in pixels. Takes effect on the next frame.
	/// Clamped to `0...maxRadius`.
	public var radius: Int {
		get { lock.withLock { shared.radius } }
		set { lock.withLock { shared.radius = Self.clampRadius(newValue) } }
	}

	// MARK: - Selection

	/// Which meshes in a selected entity's hierarchy are outlined.
	public enum MeshSelection: Sendable, Equatable {
		/// The entity's own mesh and every mesh below it. The default.
		case subtree
		/// The entity's own mesh if it has one; otherwise the nearest meshes
		/// below it. Avoids outlining a whole imported hierarchy when a
		/// grouping entity also carries a mesh.
		case nearestMeshes
		/// Meshes below the entity, falling back to its own mesh only when it
		/// has no descendant meshes. For USD stages where a parent prim carries
		/// an aggregate mesh that doesn't match the displayed child meshes.
		case preferDescendants
	}

	/// Whether extracted mesh geometry is cached per entity (default `true`).
	///
	/// Caching makes repeat selections ~1 ms instead of a fresh mesh read, but
	/// a mesh edited in place with `MeshResource.replace(with:)` that keeps its
	/// bounds keeps its old outline. Turn it off for content that's edited in
	/// place, such as a live USD stage.
	public var cachesGeometry: Bool {
		get { lock.withLock { shared.cachesGeometry } }
		set { lock.withLock { shared.cachesGeometry = newValue } }
	}

	/// Which meshes `setSelection` outlines. Applies to the next selection.
	public var meshSelection: MeshSelection {
		get { lock.withLock { shared.meshSelection } }
		set { lock.withLock { shared.meshSelection = newValue } }
	}

	/// Outlines `entity` (or clears the outline when `nil`).
	///
	/// Walks the entity subtree, packs position data and queues it for GPU
	/// upload. Geometry is read once here; movement of the selected entities is
	/// picked up by `updateTransforms()`.
	@MainActor
	public func setSelection(_ entity: Entity?) {
		setSelection(entity.map { [$0] } ?? [])
	}

	/// Outlines all of `entities` (or clears the outline when empty).
	@MainActor
	public func setSelection(_ entities: [Entity]) {
		let mode = meshSelection
		let cache = cachesGeometry ? geometryCache : OutlineGeometryCache()
		let meshes = entities.flatMap {
			OutlineMeshExtraction.collect(from: $0, mode: mode, cache: cache)
		}
		let matrices = meshes.map(\.selectionModelMatrix)
		lock.withLock {
			shared.selection = meshes
			shared.pendingUpload = meshes.map(\.data)
			shared.modelMatrices = matrices
		}
	}

	/// Reads and caches the geometry of `entities` without selecting them.
	///
	/// Reading a mesh the first time can take ~100 ms for a large model, which
	/// shows as a hitch on the first selection. Call this after loading
	/// selectable entities to move that cost to load time.
	/// Geometry is cached per entity by this renderer and re-read when the
	/// entity's mesh is swapped or its bounds change. An in-place
	/// `MeshResource.replace(with:)` edit that keeps the same bounds isn't
	/// detected.
	@MainActor
	public func preloadGeometry(of entities: [Entity]) {
		guard cachesGeometry else { return }
		let mode = meshSelection
		for entity in entities {
			_ = OutlineMeshExtraction.collect(from: entity, mode: mode, cache: geometryCache)
		}
	}

	/// Re-reads world transforms of the selected entities.
	/// Call once per frame (`SceneEvents.Update`) so moving entities stay outlined.
	@MainActor
	public func updateTransforms() {
		// Only @MainActor mutates `selection`, so this snapshot can't go stale
		// before the write below.
		let selection = lock.withLock { shared.selection }
		guard !selection.isEmpty else { return }
		let matrices = selection.map(\.currentModelMatrix)
		lock.withLock { shared.modelMatrices = matrices }
	}

	/// Camera view matrix (inverse of the camera's world transform).
	/// Call once per frame to keep the mask projection aligned with the camera.
	public func setViewMatrix(_ matrix: simd_float4x4) {
		lock.withLock { shared.viewMatrix = matrix }
	}

	// MARK: - Pipeline compilation

	/// Builds the Metal pipelines.
	/// Called from `ARView.RenderCallbacks.prepareWithDevice` or
	/// `PostProcessEffect.prepare(for:)`.
	public func prepare(device: any MTLDevice) {
		renderLock.withLock { buildPipelines(device: device) }
	}

	// Requires renderLock.
	private func buildPipelines(device: any MTLDevice) {
		render = RenderState(device: device, gpuMeshes: [])
		// Force re-upload of the current selection for the new device.
		lock.withLock {
			if shared.pendingUpload == nil {
				shared.pendingUpload = shared.selection.map(\.data)
			}
		}

		guard let library = Self.makeLibrary(device: device) else { return }

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
		vd.layouts[0].stride = MemoryLayout<SIMD3<Float>>.stride
		maskDesc.vertexDescriptor = vd
		do {
			render.maskPipeline = try device.makeRenderPipelineState(descriptor: maskDesc)
		} catch {
			Self.logger.error("Failed to build mask pipeline: \(error.localizedDescription, privacy: .public)")
		}

		render.dilatePipeline = Self.makeComputePipeline(
			"outlineDilate",
			library: library,
			device: device
		)
		render.compositePipeline = Self.makeComputePipeline(
			"outlineComposite",
			library: library,
			device: device
		)
	}

	/// Loads the package's compiled shaders, falling back to the main bundle's default library.
	private static func makeLibrary(device: any MTLDevice) -> (any MTLLibrary)? {
		do {
			return try device.makeDefaultLibrary(bundle: .module)
		} catch {
			logger.error("Failed to load shader library from package bundle: \(error.localizedDescription, privacy: .public). Falling back to the default library.")
		}
		if let library = device.makeDefaultLibrary() {
			return library
		}
		logger.error("No Metal library containing the outline shaders was found. Outlines are disabled.")
		return nil
	}

	private static func makeComputePipeline(
		_ name: String,
		library: any MTLLibrary,
		device: any MTLDevice
	) -> (any MTLComputePipelineState)? {
		guard let function = library.makeFunction(name: name) else {
			logger.error("Shader function \(name, privacy: .public) not found in library.")
			return nil
		}
		do {
			return try device.makeComputePipelineState(function: function)
		} catch {
			logger.error("Failed to build \(name, privacy: .public) pipeline: \(error.localizedDescription, privacy: .public)")
			return nil
		}
	}

	// MARK: - Encoding

	/// Encodes the three outline passes from `source` into `target`, or a plain
	/// blit when nothing is selected (or the pipelines are unavailable).
	public func encode(
		device: any MTLDevice,
		commandBuffer: any MTLCommandBuffer,
		source: any MTLTexture,
		target: any MTLTexture,
		projection: simd_float4x4
	) {
		renderLock.withLock {
			encodeLocked(
				device: device,
				cb: commandBuffer,
				source: source,
				target: target,
				projection: projection
			)
		}
	}

	// Requires renderLock.
	private func encodeLocked(
		device: any MTLDevice,
		cb: any MTLCommandBuffer,
		source: any MTLTexture,
		target: any MTLTexture,
		projection: simd_float4x4
	) {
		// Lazily build pipelines if prepare wasn't called (or the device changed).
		if render.device !== device {
			buildPipelines(device: device)
		}

		// Snapshot everything shared with other threads in one critical section.
		let frame = lock.withLock {
			let snapshot = (
				upload: shared.pendingUpload,
				modelMatrices: shared.modelMatrices,
				viewMatrix: shared.viewMatrix,
				color: shared.color,
				radius: shared.radius
			)
			shared.pendingUpload = nil
			return snapshot
		}

		if let upload = frame.upload {
			render.gpuMeshes = upload.map { Self.makeGPUMesh($0, device: device) }
		}

		// Pair each uploaded mesh with its current model matrix.
		let drawables: [(GPUMesh, simd_float4x4)] = zip(render.gpuMeshes, frame.modelMatrices)
			.compactMap { mesh, matrix in
				guard let mesh, let matrix else { return nil }
				return (mesh, matrix)
			}

		guard
			!drawables.isEmpty,
			let maskPipeline = render.maskPipeline,
			let dilatePipeline = render.dilatePipeline,
			let compositePipeline = render.compositePipeline
		else {
			blit(from: source, to: target, cb: cb)
			return
		}

		// Resolve the output first, so a failure falls back before any encoder
		// is open.
		guard let output = Self.writableTarget(target, device: device) else {
			blit(from: source, to: target, cb: cb)
			return
		}

		let w = source.width
		let h = source.height

		guard
			let (maskTex, depthTex, edgeTex) = intermediateTextures(
				device: device,
				queue: cb.commandQueue,
				w: w,
				h: h
			)
		else {
			blit(from: source, to: target, cb: cb)
			return
		}

		// Pass 1 — render mesh silhouettes into the mask texture.
		// No depth-stencil state is set, so the scene is not depth-tested:
		// the outline is X-ray and shows through occluders.
		let rpd = MTLRenderPassDescriptor()
		rpd.colorAttachments[0].texture = maskTex
		rpd.colorAttachments[0].loadAction = .clear
		rpd.colorAttachments[0].storeAction = .store
		rpd.depthAttachment.texture = depthTex
		rpd.depthAttachment.loadAction = .clear
		rpd.depthAttachment.storeAction = .dontCare
		rpd.depthAttachment.clearDepth = 1.0
		// On any encoder failure, still write the frame so it isn't dropped.
		guard let renc = cb.makeRenderCommandEncoder(descriptor: rpd) else {
			blit(from: source, to: target, cb: cb)
			return
		}
		renc.setRenderPipelineState(maskPipeline)
		let viewProjection = projection * frame.viewMatrix
		for (mesh, modelMatrix) in drawables {
			var mvp = viewProjection * modelMatrix
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
		guard let cenc1 = cb.makeComputeCommandEncoder() else {
			blit(from: source, to: target, cb: cb)
			return
		}
		cenc1.setComputePipelineState(dilatePipeline)
		cenc1.setTexture(maskTex, index: 0)
		cenc1.setTexture(edgeTex, index: 1)
		var r = Int32(frame.radius)
		cenc1.setBytes(&r, length: MemoryLayout<Int32>.size, index: 0)
		dispatch(cenc1, w: w, h: h)
		cenc1.endEncoding()

		// Pass 3 — composite the outline color over the source frame.
		guard let cenc2 = cb.makeComputeCommandEncoder() else {
			blit(from: source, to: target, cb: cb)
			return
		}
		cenc2.setComputePipelineState(compositePipeline)
		cenc2.setTexture(source, index: 0)
		cenc2.setTexture(edgeTex, index: 1)
		cenc2.setTexture(output.texture, index: 2)
		var c = Self.linear(fromSRGB: frame.color)
		cenc2.setBytes(&c, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
		var encodeSRGB: UInt32 = output.encodeSRGB ? 1 : 0
		cenc2.setBytes(&encodeSRGB, length: MemoryLayout<UInt32>.size, index: 1)
		dispatch(cenc2, w: w, h: h)
		cenc2.endEncoding()
	}

	// MARK: - Helpers

	private static func makeGPUMesh(_ data: OutlineMeshData, device: any MTLDevice) -> GPUMesh? {
		guard
			let vb = device.makeBuffer(
				bytes: data.positions,
				length: data.positions.count * MemoryLayout<SIMD3<Float>>.stride,
				options: .storageModeShared
			),
			let ib = device.makeBuffer(
				bytes: data.indices,
				length: data.indices.count,
				options: .storageModeShared
			)
		else {
			logger.error("Failed to allocate outline mesh buffers.")
			return nil
		}
		return GPUMesh(
			vertexBuffer: vb,
			indexBuffer: ib,
			indexCount: data.indexCount,
			indexType: data.indexType
		)
	}

	/// sRGB-encoded RGBA → linear RGB (alpha unchanged). Extended-range
	/// components (below 0 or above 1, from wide-gamut colors) are kept, using
	/// the sign-mirrored transfer function of extended sRGB.
	static func linear(fromSRGB c: SIMD4<Float>) -> SIMD4<Float> {
		func channel(_ v: Float) -> Float {
			let a = abs(v)
			let l = a <= 0.04045 ? a / 12.92 : pow((a + 0.055) / 1.055, 2.4)
			return v < 0 ? -l : l
		}
		return SIMD4(channel(c.x), channel(c.y), channel(c.z), c.w)
	}

	/// GPUs without `.apple2` support can't write sRGB textures from a shader.
	/// Following Apple's guidance, an 8-bit sRGB target is written through a
	/// non-sRGB view, with the composite pass encoding linear→sRGB itself
	/// (`encodeSRGB`) so the frame doesn't darken. Other formats (RealityKit
	/// also hands out `rgba16Float` targets) are written directly; creating a
	/// view of an incompatible format would abort instead of returning nil.
	/// https://developer.apple.com/documentation/realitykit/checking-the-pixel-format-of-a-postprocess-effect-s-output-texture
	private static func writableTarget(
		_ target: any MTLTexture,
		device: any MTLDevice
	) -> (texture: any MTLTexture, encodeSRGB: Bool)? {
		if device.supportsFamily(.apple2) { return (target, false) }

		let viewFormat: MTLPixelFormat
		switch target.pixelFormat {
		case .bgra8Unorm_srgb: viewFormat = .bgra8Unorm
		case .rgba8Unorm_srgb: viewFormat = .rgba8Unorm
		default: return (target, false)
		}
		// Switching only between sRGB and linear variants of the same layout
		// doesn't need `.pixelFormatView` usage.
		guard let view = target.makeTextureView(pixelFormat: viewFormat) else {
			logger.error("Can't write the sRGB output texture on this GPU; outline skipped.")
			return nil
		}
		return (view, true)
	}

	/// Returns `queue`'s cached mask/depth/edge textures, recreating them only
	/// when the size changes.
	// Requires renderLock.
	private func intermediateTextures(
		device: any MTLDevice,
		queue: any MTLCommandQueue,
		w: Int,
		h: Int
	) -> (any MTLTexture, any MTLTexture, any MTLTexture)? {
		let key = ObjectIdentifier(queue)
		if let cached = render.intermediates[key], cached.queue === queue,
			cached.mask.width == w, cached.mask.height == h
		{
			return (cached.mask, cached.depth, cached.edge)
		}

		guard
			let mask = makeTexture(
				device: device,
				w: w,
				h: h,
				format: .r8Unorm,
				usage: [.renderTarget, .shaderRead]
			),
			let depth = makeTexture(
				device: device,
				w: w,
				h: h,
				format: .depth32Float,
				usage: .renderTarget
			),
			let edge = makeTexture(
				device: device,
				w: w,
				h: h,
				format: .r8Unorm,
				usage: [.shaderRead, .shaderWrite]
			)
		else {
			Self.logger.error("Failed to allocate \(w)x\(h) outline textures.")
			return nil
		}
		render.intermediates = render.intermediates.filter { $0.value.queue != nil }
		render.intermediates[key] = IntermediateTextures(queue: queue, mask: mask, depth: depth, edge: edge)
		return (mask, depth, edge)
	}

	private func dispatch(
		_ encoder: any MTLComputeCommandEncoder,
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

	/// Converts a `CGColor` to sRGB RGBA components (yellow if conversion fails).
	private static func rgba(from cgColor: CGColor) -> SIMD4<Float> {
		guard
			let srgb = CGColorSpace(name: CGColorSpace.sRGB),
			let converted = cgColor.converted(to: srgb, intent: .defaultIntent, options: nil),
			let c = converted.components, c.count >= 4
		else {
			logger.error("Could not convert outline color to sRGB; using yellow.")
			return [1, 1, 0, 1]
		}
		return SIMD4(Float(c[0]), Float(c[1]), Float(c[2]), Float(c[3]))
	}
}

#include <memory>								// std::unique_ptr

#include "cuda_texture_types.h"					// texture
#include "texture_fetch_functions.h"			// tex1Dfetch

#include "cuda_includes.h"
#include "attachment.cuh"

namespace thin
{
	namespace attach
	{
		namespace _private
		{
			// Type of all the device texture references in this module.
			typedef texture<DFaceMaskType, cudaTextureType1D, cudaReadModeElementType> MaskTexType;
			// Number of neighbor offsets
			const unsigned NUM_NB_OFFSET = 26U;
			// A singleton class that unions all the device pointers for texture reference.
			class DevArrPtrs
			{
			public:
				static DevArrPtrs* instance()
				{
					if (!m_instance)
					{
						m_instance = std::unique_ptr<DevArrPtrs>(new DevArrPtrs);
					}
					return m_instance.get();
				}

				DFaceMaskType* d_packedFaceMaskArr;
			private:
				static std::unique_ptr<DevArrPtrs> m_instance;
			};

			std::unique_ptr<DevArrPtrs> DevArrPtrs::m_instance = nullptr;
			// Total number of the array storing face masks.
			const unsigned PACKED_FACE_MASK_ARR_SIZE = 44U;
			// The beginning index of the low face masks from a 2-face to 1-faces.
			const unsigned D2ToD1TexBegin = 0;
			// The begining index of the low face masks from a 2-face to 0-faces.
			const unsigned D2ToD0TexBegin = 6U;
			// The begining index of the low face masks from a 1-face to 0-faces.
			const unsigned D1ToD0TexBegin = 12U;
			// The begining index of the high face masks from a 1-face to 2-faces.
			const unsigned D1ToD2TexBegin = 24U;
			// The begining index of the high face masks from a 0-face to 1-faces.
			const unsigned D0ToD1TexBegin = 36U;
			// Texture data that holds all the necessary d-face masks.
			const DFaceMaskType h_packedFaceMaskArr[PACKED_FACE_MASK_ARR_SIZE] =
			{
				// Dim 2 to Dim 1
				// range: [0, 5], size: 6
				(1 << 0) | (1 << 1) | (1 << 2) | (1 << 3),
				(1 << 0) | (1 << 4) | (1 << 5) | (1 << 8),
				(1 << 1) | (1 << 5) | (1 << 6) | (1 << 9),
				(1 << 2) | (1 << 6) | (1 << 7) | (1 << 10),
				(1 << 3) | (1 << 4) | (1 << 7) | (1 << 11),
				(1 << 8) | (1 << 9) | (1 << 10) | (1 << 11),
				// Dim 2 to Dim 0
				// range: [6, 11], size: 6
				(1 << 0) | (1 << 1) | (1 << 2) | (1 << 3),
				(1 << 0) | (1 << 1) | (1 << 4) | (1 << 5),
				(1 << 1) | (1 << 2) | (1 << 5) | (1 << 6),
				(1 << 2) | (1 << 3) | (1 << 6) | (1 << 7),
				(1 << 0) | (1 << 3) | (1 << 4) | (1 << 7),
				(1 << 4) | (1 << 5) | (1 << 6) | (1 << 7),
				// Dim 1 to Dim 0
				// range: [12, 23], size: 12
				(1 << 0) | (1 << 1),
				(1 << 1) | (1 << 2),
				(1 << 2) | (1 << 3),
				(1 << 0) | (1 << 3),
				(1 << 0) | (1 << 4),
				(1 << 1) | (1 << 5),
				(1 << 2) | (1 << 6),
				(1 << 3) | (1 << 7),
				(1 << 4) | (1 << 5),
				(1 << 5) | (1 << 6),
				(1 << 6) | (1 << 7),
				(1 << 4) | (1 << 7),
				// Dim 1 to Dim 2
				// range: [24, 35], size: 12
				(1 << 0) | (1 << 1),
				(1 << 0) | (1 << 2),
				(1 << 0) | (1 << 3),
				(1 << 0) | (1 << 4),
				(1 << 1) | (1 << 4),
				(1 << 1) | (1 << 2),
				(1 << 2) | (1 << 3),
				(1 << 3) | (1 << 4),
				(1 << 1) | (1 << 5),
				(1 << 2) | (1 << 5),
				(1 << 3) | (1 << 5),
				(1 << 4) | (1 << 5),
				// Dim 0 to Dim 1
				// range: [36, 43], size: 8
				(1 << 0) | (1 << 3)| (1 << 4),
				(1 << 0) | (1 << 1)| (1 << 5),
				(1 << 1) | (1 << 2)| (1 << 6),
				(1 << 2) | (1 << 3)| (1 << 7),
				(1 << 4) | (1 << 8)| (1 << 11),
				(1 << 5) | (1 << 8)| (1 << 9),
				(1 << 6) | (1 << 9)| (1 << 10),
				(1 << 7) | (1 << 10)| (1 << 11)
			};
			// Device texture reference for all d-face bit masks.
			MaskTexType packedFaceMaskTex;
			// Initialize the device texture references of this module.
			void _initDeviceTes(DFaceMaskType** d_packedFaceMaskArr)
			{
				const cudaChannelFormatDesc uint16Desc = cudaCreateChannelDesc(8 * sizeof(DFaceMaskType), 0, 0, 0, cudaChannelFormatKindUnsigned);

				checkCudaErrors(cudaMalloc(d_packedFaceMaskArr, sizeof(DFaceMaskType) * PACKED_FACE_MASK_ARR_SIZE));
				checkCudaErrors(cudaMemcpy(*d_packedFaceMaskArr, h_packedFaceMaskArr, sizeof(DFaceMaskType) * PACKED_FACE_MASK_ARR_SIZE, cudaMemcpyHostToDevice));
				checkCudaErrors(cudaBindTexture(0, packedFaceMaskTex, *d_packedFaceMaskArr, uint16Desc, sizeof(DFaceMaskType) * PACKED_FACE_MASK_ARR_SIZE));
			}
			// Unbinds the GPU texture references and frees the device memory.
			void _clearDeviceTex(DFaceMaskType* d_packedFaceMaskArr)
			{
				checkCudaErrors(cudaFree(d_packedFaceMaskArr));
			}
		}; // namespace thin::attach::_private

		namespace ap = thin::attach::_private;
		namespace tp = thin::_private;

		void initDevice()
		{
			ap::DevArrPtrs* ptrs = ap::DevArrPtrs::instance();

			ap::_initDeviceTes(&(ptrs->d_packedFaceMaskArr));
			ap::_initDeviceTes(&(ptrs->d_packedFaceMaskArr));
		}

		void shutdownDevice()
		{
			ap::DevArrPtrs* ptrs = ap::DevArrPtrs::instance();

			ap::_clearDeviceTex(ptrs->d_packedFaceMaskArr);
		}

		__device__ DFaceMaskType D2ManagerPolicy::getHiMask(DFaceIndexType faceIdx)
		{
			// This function only implements the interface and should never be called.
			return 0;
		}

		__device__ DFaceMaskType D2ManagerPolicy::getLowMask(DFaceIndexType faceIdx)
		{
			return tex1Dfetch(ap::packedFaceMaskTex, ap::D2ToD1TexBegin + faceIdx);
		}

		__device__ DFaceMaskType D1ManagerPolicy::getHiMask(DFaceIndexType faceIdx)
		{
			return tex1Dfetch(ap::packedFaceMaskTex, ap::D1ToD2TexBegin + faceIdx);
		}

		__device__ DFaceMaskType D1ManagerPolicy::getLowMask(DFaceIndexType faceIdx)
		{
			return tex1Dfetch(ap::packedFaceMaskTex, ap::D1ToD0TexBegin + faceIdx);
		}

		__device__ DFaceMaskType D0ManagerPolicy::getHiMask(DFaceIndexType faceIdx)
		{
			return tex1Dfetch(ap::packedFaceMaskTex, ap::D0ToD1TexBegin + faceIdx);
		}

		__device__ DFaceMaskType D0ManagerPolicy::getLowMask(DFaceIndexType faceIdx)
		{
			// This function should never be called.
			return 0;
		}

		__device__ DFaceMaskType D2ToD0MaskPolicy::getMask(DFaceIndexType faceIdx)
		{
			return tex1Dfetch(ap::packedFaceMaskTex, ap::D2ToD0TexBegin + faceIdx);
		}

		////////// Implementation of Attachment class //////////
		__device__ void Attachment::setFace(DFaceIndexType faceIdx)
		{
			// Set the literal face
			_setFaceOnly(faceIdx);
			// Set the edges contained by this literal face.

			// To-DO:
			// Use setEdge instead of _setEdgeOnly, the former will take
			// care of the vertices that should be set internally.
			DFaceMaskType edgeMask = m_faceMngr.getLowDimFacesMask(faceIdx);
			for (uint8_t edgeIdx = 0; edgeIdx < m_edgeMngr.numFaces(); ++edgeIdx)
			{
				if (tp::_readBit(edgeMask, edgeIdx))
				{
					_setEdgeOnly(edgeIdx);
				}
			}

			DFaceMaskType vertexMask = D2ToD0MaskPolicy::getMask(faceIdx);
			for (unsigned vertexIdx = 0; vertexIdx < m_vertexMngr.numFaces(); ++vertexIdx)
			{
				if (tp::_readBit(vertexMask, vertexIdx))
				{
					_setVertexOnly(vertexIdx);
				}
			}
		}

		__device__ void Attachment::setEdge(DFaceIndexType edgeIdx)
		{
			// Set the edge
			_setEdgeOnly(edgeIdx);

			// Set the vertices contained by this edge.
			DFaceMaskType vertexMask = m_edgeMngr.getLowDimFacesMask(edgeIdx);
			for (unsigned vertexIdx = 0; vertexIdx < m_vertexMngr.numFaces(); ++vertexIdx)
			{
				if (tp::_readBit(vertexMask, vertexIdx))
				{
					_setVertexOnly(vertexIdx);
				}
			}
		}

		__device__ void Attachment::setVertex(DFaceIndexType vertexIdx)
		{
			_setVertexOnly(vertexIdx);
		}

		__device__ bool Attachment::isCollapsible()
		{
			_collapse();

			// If there are still literal faces or edges left, then
			// it is not collapsible.
			if (m_faceMngr.numSetFaces() || m_edgeMngr.numSetFaces())
			{
				return false;
			}
			// At this point, it's collapsible iff only 1 vertex left.
			return m_vertexMngr.numSetFaces() == 1;
		}

		__device__ void Attachment::_setFaceOnly(DFaceIndexType faceIdx)
		{
			m_faceMngr.set(faceIdx);
		}

		__device__ void Attachment::_setEdgeOnly(DFaceIndexType edgeIdx)
		{
			m_edgeMngr.set(edgeIdx);
		}

		__device__ void Attachment::_setVertexOnly(DFaceIndexType vertexIdx)
		{
			m_vertexMngr.set(vertexIdx);
		}

		__device__ void Attachment::_collapseFaces()
		{
			_collapseDimFaces(m_faceMngr, m_edgeMngr);
		}

		__device__ void Attachment::_collapseEdges()
		{
			_collapseDimFaces(m_edgeMngr, m_vertexMngr);
		}

		__device__ void Attachment::_collapse()
		{
			_collapseFaces();
			// We don't need to continue the collapsing process if
			// literal faces still exist.
			if (m_faceMngr.numSetFaces()) return;

			_collapseEdges();
		}
		// Set the attachment @attach according to the index of the neighborhood offset
		// @offsetIndex.
		//
		// This function is coupled with the order of neighbor offsets in "neighbor.cu".
		// The coupling is worthy in that it makes the if-else stmt to be really simple,
		// which reduces the checking branches.
		//
		// [precondition] 0 <= @offsetIndex < NUM_NB_OFFSET
		__device__ void _setAttachmentByOffsetIndex(Attachment& attach, uint8_t offsetIndex)
		{
			if (offsetIndex < NUM_VERTICES)
			{
				attach.setVertex(offsetIndex);
			}
			else
			{
				offsetIndex -= NUM_VERTICES;
				if (offsetIndex < NUM_EDGES)
				{
					attach.setEdge(offsetIndex);
				}
				else
				{
					offsetIndex -= NUM_EDGES;
					attach.setFace(offsetIndex);
				}
			}
		}

		__device__ Attachment
		generateAttachment(const nb::NbMaskType nbMask)
		{
			Attachment attach;

			for (uint8_t offsetIndex = 0; offsetIndex < ap::NUM_NB_OFFSET; ++offsetIndex)
			{
				if (tp::_readBit(nbMask, offsetIndex))
				{
					_setAttachmentByOffsetIndex(attach, offsetIndex);
				}
			}

			return attach;
		}
	}; // namespace thin::attach;
}; // namespace thin;

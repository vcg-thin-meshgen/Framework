#ifndef ATTACHMENT_CUH
#define ATTACHMENT_CUH

// This file defines the Attachment class and associated utilities
// for checking if a voxel is Simple or not. A voxel is simple
// when its neighbor voxels (excluding itself) is Reducible.
//
// Users do not need to touch this file.

#include "cuda_includes.h"
#include "thinning_base.cuh"
#include "neighbor.cuh"

namespace thin
{
	namespace attach
	{
		// A d-face can be a voxel(d=3), a face(d=2), an edge(d=1) or a vertex(d=0),
		// depending on the dimension d. Each d-face could contain multiple d'-faces,
		// where d' < d. The set of d'-faces contained by a d-face is essential for this
		// algorithm to work. Equivalently, a d-face could be contained by multiple
		// d'-faces, where d' > d. The set of d'faces containing d-face is also
		// essential.
		//
		// It is enough to use a 16-bit uint to represent such sets, because a voxel
		// contains only 6 faces, 12 edges and 8 vertices. And max{6, 12, 8} < 16. If we
		// label the d-faces according to a certain order, then each bit in the mask
		// will represent one of such d-faces. In this project, the label strategy of
		// the d-faces is shown at the beginning of "thin_base.cuh".
		typedef uint16_t DFaceMaskType;
		// constants definition
		const unsigned NUM_FACES = 6U;
		const unsigned NUM_EDGES = 12U;
		const unsigned NUM_VERTICES = 8U;

		// DFaceManager is used to manage the d-faces information at the same dimension.
		// It records which d-faces exist in the Attachment, which we will explain later
		// on. Internally, it also uses the same bitmap strategy to achieve the
		// recording of existence.
		//
		// NUM stores the number of total d-faces at this dimension. Different DimPolicy
		// provides unqiue data for different dimension d.
		template <uint8_t NUM, typename DimPolicy>
		class DFaceManager
		{
		public:
			__device__
			DFaceManager() : m_bits(0) { }
			// Set the existence of a d-face according to @faceIdx.
			__device__ void set(DFaceIndexType faceIdx)
			{
				using namespace thin::_private;
				_setBit(m_bits, faceIdx);
			}
			// Clear the existence of a d-face according to @faceIdx.
			__device__ void clear(DFaceIndexType faceIdx)
			{
				using namespace thin::_private;
				_clearBit(m_bits, faceIdx);
			}
			// Check if a d-face, indicated by @faceIdx, exists.
			__device__ bool isSet(DFaceIndexType faceIdx) const
			{
				using namespace thin::_private;
				return (bool)_readBit(m_bits, faceIdx);
			}
			// Return the bit mask of d-faces that exists.
			__device__ DFaceMaskType findSetFacesMask() const
			{
				return m_bits;
			}
			// Total number of d-faces at this dimension.
			__device__ uint8_t numFaces() const
			{
				return NUM;
			}
			// Existing number of d-faces at this dimension.
			__device__ uint8_t numSetFaces() const
			{
				using namespace thin::_private;
				return _countNumSetBits(m_bits, NUM);
			}
			// Get the bit mask of all d'-faces at a higher dim, d', that contains the
			// d-face indicated by @faceIdx.
			__device__ DFaceMaskType getHiDimFacesMask(DFaceIndexType faceIdx) const
			{
				return DimPolicy::getHiMask(faceIdx);
			}
			// Get the bit mask of all d'faces at a lower dim, d', that are contained by
			// the d-face indicated by @faceIdx.
			__device__ DFaceMaskType getLowDimFacesMask(DFaceIndexType faceIdx) const
			{
				return DimPolicy::getLowMask(faceIdx);
			}

		private:
			DFaceMaskType m_bits;
		};

		// Dim 2 Policy for DFaceManager
		class D2ManagerPolicy
		{
		public:
			__device__ static DFaceMaskType getHiMask(DFaceIndexType faceIdx);
			__device__ static DFaceMaskType getLowMask(DFaceIndexType faceIdx);
		};

		// Dim 1 Policy for DFaceManager
		class D1ManagerPolicy
		{
		public:
			__device__ static DFaceMaskType getHiMask(DFaceIndexType faceIdx);
			__device__ static DFaceMaskType getLowMask(DFaceIndexType faceIdx);
		};

		// Dim 0 Policy for DFaceManager
		class D0ManagerPolicy
		{
		public:
			__device__ static DFaceMaskType getHiMask(DFaceIndexType faceIdx);
			__device__ static DFaceMaskType getLowMask(DFaceIndexType faceIdx);
		};

		class D2ToD0MaskPolicy
		{
		public:
			__device__ static DFaceMaskType getMask(DFaceIndexType faceIdx);
		};

		// Initialize the device side resources for this module to work.
		void initDevice();
		// Release the device side resources for this module to work.
		void shutdownDevice();

		using FaceManager = DFaceManager<NUM_FACES, D2ManagerPolicy>;
		using EdgeManager = DFaceManager<NUM_EDGES, D1ManagerPolicy>;
		using VertexManager = DFaceManager<NUM_VERTICES, D0ManagerPolicy>;

		// An Attachment is a union of d-faces, where d could be from more than a single
		// dimension. It can be viewed as some residue "attached" to the voxel from its
		// neighbor voxels, and is a vital part to the algorithm that checks whether a
		// voxel is Simple or not.
		class Attachment
		{
		public:
			// Set the existence of a literal face (2-face)
			__device__ void setFace(DFaceIndexType faceIdx);
			// Set the existence of an edge (1-face)
			__device__ void setEdge(DFaceIndexType edgeIdx);
			// Set the existence of a vertex (0-face)
			__device__ void setVertex(DFaceIndexType vertexIdx);

			// Key function to detect if the voxel this Attachment belongs to
			// is Simple. A voxel is Simple iff its Attachment is Collapsible.
			__device__ bool isCollapsible();

		private:
			// Only set the existence of 2-face, without the d'-faces at
			// the lower dimensions.
			__device__ void _setFaceOnly(DFaceIndexType faceIdx);
			// Only set the existence of 1-face, without the d'-faces at
			// the lower dimensions.
			__device__ void _setEdgeOnly(DFaceIndexType edgeIdx);
			// Only set the existence of 0-face.
			__device__ void _setVertexOnly(DFaceIndexType vertexIdx);

			__device__ void _collapseFaces();
			__device__ void _collapseEdges();
			// Collapse the entire attachment until stability.
			__device__ void _collapse();

			// Find the (d+1)-faces that share the d-face given @bylowDimFaceIdx and
			// return the bit mask of such (d+1)-faces.
			//
			// [precondition] @lowDimMngr managers the d-face information and @hiDimMngr
			// managers the (d+1)-face information.
			template <typename LOW, typename HI>
			__device__ DFaceMaskType
			_lowDimFacesSharedBy(DFaceIndexType lowDimFaceIdx, const LOW& lowDimMngr, const HI& hiDimMngr) const
			{
				// The bitmap with all the (d+1)-faces that contains the d-face at
				// @lowDimFaceIdx set to 1.
				DFaceMaskType hiDimFacesMask = lowDimMngr.getHiDimFacesMask(lowDimFaceIdx);
				// The bitmap with all the existing (d+1)-faces set to 1.
				DFaceMaskType hiDimSetFacesMask = hiDimMngr.findSetFacesMask();

				return hiDimFacesMask & hiDimSetFacesMask;
			}

			// Generic function to collapse d-faces at dimension d implicitly given by
			// @hiDimMngr. This will require the corporation between DFaceManagers at both
			// dimension d and (d-1).
			//
			// [precondition] @lowDimMngr managers the d-face information and @hiDimMngr
			// managers the (d+1)-face information.
			template <typename HI, typename LOW>
			__device__ void _collapseDimFaces(HI& hiDimMngr, LOW& lowDimMngr)
			{
				using namespace thin::_private;
				uint8_t lastIterSize = hiDimMngr.numSetFaces();

				while (lastIterSize)
				{
					// Loop through all the set d-faces.
					DFaceMaskType hiDimSetFacesMask = hiDimMngr.findSetFacesMask();
					bool noFreePairFound = true;

					for (uint8_t hiDimFaceIdx = 0;
						 (hiDimFaceIdx < hiDimMngr.numFaces()) && noFreePairFound;
						 ++hiDimFaceIdx)
					{
						// Check if hiDimFaceIdx-th d-face exists.
						if (_readBit(hiDimSetFacesMask, hiDimFaceIdx))
						{
							// If it does, read out the bit mask of all the (d-1)-faces
							// this d-face could possibly contain.
							DFaceMaskType lowDimFaceMask = hiDimMngr.getLowDimFacesMask(hiDimFaceIdx);

							for (uint8_t lowDimFaceIdx = 0;
								 (lowDimFaceIdx < lowDimMngr.numFaces()) && noFreePairFound;
								 ++lowDimFaceIdx)
							{
								// Check if any of the (d-1)-faces in lowDimFaceMask exists.
								if (_readBit(lowDimFaceMask, lowDimFaceIdx))
								{
									// If the (d-1)-face indeed exists, find out all the d-faces
									// that are currently containg this (d-1)-face.
									DFaceMaskType lowDimFacesSharedByMask = _lowDimFacesSharedBy(lowDimFaceIdx, lowDimMngr, hiDimMngr);
									if (_countNumSetBits(lowDimFacesSharedByMask, hiDimMngr.numFaces()) == 1)
									{
										// Only 1 d-face contains it, this d-face and (d-1)-face constitues
										// a free pair that can be safely removed.
										hiDimMngr.clear(hiDimFaceIdx);
										lowDimMngr.clear(lowDimFaceIdx);

										noFreePairFound = false;
										break;
									}
								}
							}
						}
					}
					// Update the size, or the number of set faces.
					uint8_t curIterSize = hiDimMngr.numSetFaces();
					if (curIterSize == lastIterSize) break;

					lastIterSize = curIterSize;
				}
			}

			FaceManager m_faceMngr;
			EdgeManager m_edgeMngr;
			VertexManager m_vertexMngr;
		};

		// Generate the attachment object for the given @nbMask.
		__device__ Attachment
		generateAttachment(const nb::NbMaskType nbMask);
	}; // namespace thin::attach;
}; // namespace thin;
#endif

import numpy as np
from numpy import linalg as LA

if __name__ == '__main__':
	label = 320
	ifname, ofname = '{0}.obj'.format(label), '{0}.pwn'.format(label)

	vertices = []
	surfacePts, surfaceNs = [], []
	
	print 'reading data...'
	linecount = 0
	with open(ifname, 'r') as ifh:
		for line in ifh:
			if linecount % 50000 == 0:
				print 'reading data up to line {0}'.format(linecount)

			linecount += 1

			line = line.strip()
			tok = line[:1]
			line = line[2:].split(' ')
			
			if tok == 'v':
				vtx = np.array([float(f) for f in line])
				vertices.append(vtx)

			elif tok == 'f':
				vidx = [int(s) - 1 for s in line]
				va, vb, vc = vertices[vidx[0]], vertices[vidx[1]], vertices[vidx[2]]

				pt = (va + vb + vc) / 3.0
				nm = np.cross(vb - va, vc - va)
				if LA.norm(nm) > 1e-10:
					nm = nm / LA.norm(nm)

					surfacePts.append(pt)
					surfaceNs.append(nm)

			else:
				raise Exception('unkonw token: {0}'.format(tok))

	assert len(surfacePts) == len(surfaceNs)

	print 'total pts: {0}, converting to .pwn'.format(len(surfacePts))

	with open(ofname, 'w') as ofh:
		ofh.write('{0}\n'.format(len(surfacePts)))

		for pt in surfacePts:
			line = '{0} {1} {2}\n'.format(pt[0], pt[1], pt[2])
			ofh.write(line)

		for nm in surfaceNs:
			line = '{0} {1} {2}\n'.format(nm[0], nm[1], nm[2])
			ofh.write(line)			
// Bounding boxes of the connected parts of one label. No DOM, no three.js: a node test can import it.
//
// `lab` holds the labels in the order (i * ny + j) * nz + k. Two voxels are connected if they share a face.
// A load region and a solid region (`regions.solid`) both have the label 3. If they are apart, each part gets its
// own box. If they touch, they are one part and have one box.
// Return [[i0, j0, k0, i1, j1, k1], ...] (voxel indices, both ends included), the part with the first voxel first.
export function componentBoxes(lab, nx, ny, nz, code) {
  const seen = new Uint8Array(lab.length);
  const out = [];
  const stack = [];
  const nyz = ny * nz;
  for (let s = 0; s < lab.length; s++) {
    if (lab[s] !== code || seen[s]) continue;
    const box = [nx, ny, nz, -1, -1, -1];
    seen[s] = 1;
    stack.push(s);
    while (stack.length) {
      const p = stack.pop();
      const i = Math.floor(p / nyz), j = Math.floor(p / nz) % ny, k = p % nz;
      if (i < box[0]) box[0] = i;
      if (j < box[1]) box[1] = j;
      if (k < box[2]) box[2] = k;
      if (i > box[3]) box[3] = i;
      if (j > box[4]) box[4] = j;
      if (k > box[5]) box[5] = k;
      const visit = (q) => {
        if (lab[q] === code && !seen[q]) { seen[q] = 1; stack.push(q); }
      };
      if (i > 0) visit(p - nyz);
      if (i < nx - 1) visit(p + nyz);
      if (j > 0) visit(p - nz);
      if (j < ny - 1) visit(p + nz);
      if (k > 0) visit(p - 1);
      if (k < nz - 1) visit(p + 1);
    }
    out.push(box);
  }
  return out;
}

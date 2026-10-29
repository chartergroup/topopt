// Places of the load case labels that would lie on top of each other. No DOM, no three.js: a node test can import it.
//
// Two load cases with their arrows at the same place had their labels on top of each other. `points` are the label
// centres [[x, y, z], ...] in load case order; a label is `w` wide (one number, or one width for each label) and `h` high.
// Two labels collide if their centres are less than half the sum of their widths apart in x-y and less than `h` apart in z.
// A label that collides with an earlier one moves in z, in the direction `ways[n]` (1 up, -1 down), to just past it, until
// it collides with none. Return the move (in z) of each label; a label that collides with none has 0.
export function labelShifts(points, w, h, ways) {
  const wOf = (k) => (Array.isArray(w) ? w[k] : w);
  const placed = [];
  return points.map(([x, y, z], n) => {
    let at = z;
    for (let i = 0; i < placed.length; i++) {
      const [px, py, pz] = placed[i];
      if (Math.hypot(x - px, y - py) < (wOf(n) + wOf(i)) / 2 && Math.abs(at - pz) < h) {
        at = pz + ways[n] * h;
        i = -1;   // the new height may cover an earlier label: check all of them again
      }
    }
    placed.push([x, y, at]);
    return at - z;
  });
}

// Load cases with their arrows at the same place. The later arrow is drawn over the earlier one and hides it, so the cases share one
// label. `arrows` are [{tail: [x, y, z], dir: [x, y, z]}, ...] in load case order (`dir` a unit vector). Two arrows are at the same
// place if their tails are less than `tol` apart and their directions differ by less than about 2.6 degrees (cosine above 0.999).
// Return the groups as lists of indices, each in load case order, ordered by their first index; a case alone is a group of one.
export function sameArrowGroups(arrows, tol) {
  const groups = [];
  arrows.forEach((a, n) => {
    const g = groups.find((gr) => {
      const b = arrows[gr[0]];
      const gap = Math.hypot(a.tail[0] - b.tail[0], a.tail[1] - b.tail[1], a.tail[2] - b.tail[2]);
      return gap < tol && a.dir[0] * b.dir[0] + a.dir[1] * b.dir[1] + a.dir[2] * b.dir[2] > 0.999;
    });
    if (g) g.push(n);
    else groups.push([n]);
  });
  return groups;
}

// The centres of the labels of `groups` (from sameArrowGroups; `arrows` [{tail, dir, ci}, ...], `ci` the index of the case). A label
// is `len` * 0.4 high and `widths[g]` wide; `len` is the arrow length. It starts 0.2 len past the tail of the first arrow of its group,
// 0.12 len up (an even case index) or down (an odd one), and moves past an earlier label that it would cover, away from the tip of
// its arrow (up for a downward or level force): toward the tail, so it never lies on its own arrow.
export function labelPlaces(arrows, groups, len, widths) {
  const points = groups.map((g) => {
    const { tail, dir, ci } = arrows[g[0]];
    return [0, 1, 2].map((k) => tail[k] - dir[k] * len * 0.2 + (k === 2 ? len * 0.12 * (ci % 2 ? -1 : 1) : 0));
  });
  const ways = groups.map((g) => (arrows[g[0]].dir[2] > 0 ? -1 : 1));
  const dz = labelShifts(points, widths, len * 0.4, ways);
  return points.map(([x, y, z], i) => [x, y, z + dz[i]]);
}

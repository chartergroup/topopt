# Pictures of the README

All pictures come from our own files: the examples `examples/footbridge/` and `examples/hook/`, a recorded run of an older build
(see [recordings/README.md](../../recordings/README.md)) and the icon of our website. No picture comes from another project.

| File | What it shows | Size |
|---|---|---|
| `hero.gif` | The dashboard replay of a footbridge run (see "The footbridge clip" below): the 3D view and the numbers bar. 1116 x 1080 pixels, 10 frames each second, 23.6 s: the shape forms slowly (11 of the 200 passes, with soft fades), then one turn of the finished part. | 5.4 MB |
| `hero.mp4` | The same clip, sharper. 1116 x 1080 pixels, 30 frames each second, 23.6 s. | 0.8 MB |
| `icon.svg` | The icon of the topopt website (the cube of `https://topopt.ai/assets/favicon.svg`, our own) on a light rounded tile, in the title line of the README. The tile keeps the dark cube readable on a light and on a dark page, whatever the theme. | 0.01 MB |
| `dashboard-done.png` | The final screen of the full dashboard of the recorded hook run, with the code panel and the result, 1280 x 640 pixels (the size of a social preview). A fallback for the clip. | 0.25 MB |
| `before-after-under.png` | The hook: the design space before the run and the part after the run, view `under` (from the front and below). 1502 x 640 pixels. | 0.09 MB |
| `frame-sketch.png` | A side elevation of a bike frame with its dimensions: the sketch on the website. 1000 x 647 pixels. The README links it to the full size. | 0.2 MB |

The GIF is the first picture of the README, so it must load fast. A GIF is much bigger than a video of the same length.
The MP4 is the sharp version of the same clip (0.8 MB). The README shows only the GIF and does not link the MP4: the GIF is sharp enough at its full size.
The GIF of the footbridge is 5.4 MB, more than the 3 MB of the recipe below: it keeps the full size of the frames (no scale step).

The clip has no fast text. The code panel and the log lines of the dashboard are not in it: they change at each step and scroll,
and fast text in a clip can make a viewer feel sick. For the same reason the status label shows `RUNNING` without the name of the step.
The motion is slow too: while the part forms, the camera does not move, and the shape changes with soft fades.
`dashboard-done.png` shows the full dashboard.

## How `frame-sketch.png` was made

The source is the sketch image of the website, `https://topopt.ai/assets/early-frame-sketch.png` (1978 x 1279 pixels). The file is not cropped
and its content is not changed. It is only made smaller with Pillow (`Image.resize` to 1000 pixels wide, LANCZOS filter, saved with `optimize=True`),
so that the README loads fast. In the README the small copy links to the full size at the URL above.

## How the strips were made

The strip `before-after-under.png` shows a fresh run of `examples/hook/hook.yaml` as it is: `settings.solver: auto`, which picked `gpu-mg` on the integrated GPU of an Intel Core Ultra 9 185H host, 2026-10-28, on main 3585ef6.
The mass went from 85.6 g to 25.35 g (70.4 % less), and the run took 133.3 s.
Its tiles are `topopt render` pictures with the view `under` (from the front and below). From below, the open space between the arms of
the hook shows how much of the block the run took away; a side view (`front`) hides it, and the default view `iso` hides members behind a load plate.
The other project views are `front` (`0,-1,0`) and `fl2` (`-1.333,-1,0.373`, from the front left, a little above): the tiles of
[the examples](../../examples/README.md) use them.
Run the commands in the repo folder. `<out>` is a folder of your choice.

```
topopt run examples/hook/hook.yaml --out runs/hook

# after: the finished part, no regions drawn
topopt render runs/hook/result/final.stl --role part --regions off --size 900x600 --views "under:0.5,-1,-0.5" --out <out>/after

# before: the design space (grey), the fixed plate (blue) and the load region (red)
topopt render examples/hook/design.stl examples/hook/wall_plate.stl examples/hook/hook_tip.stl --role part,fixed,load --regions solid --size 900x600 --views "under:0.5,-1,-0.5" --out <out>/before
```

Both renders have the same camera and the same scale, so the two tiles of a strip are side by side without a change of size.
The strip is the two tiles of one view, cropped, with a caption line below each tile.

## How the clip was made

These are the steps of the hook clip of 2026-10-25. "The footbridge clip" below says what the clip of the README does in another way.

1. Start the replay of the recording with a replay clock that you set by hand, one pass at a time
   (`topopt dashboard <recorded run> --replay --port <free port> --no-open`; the `Replay` class takes a `now` function).
2. Open the page in a headless browser with a window of 1280 x 720 pixels. Replace the frame loop of the page (`requestAnimationFrame`)
   with one that you step by hand. While the part forms, the page time does not move, so the model does not sway: only the shape changes.
   For each pass, set the clock to a time late in the pass (92 % of the way to the next pass), wait until the page shows that pass, step the
   frame loop once, and take a screenshot.
3. Take one screenshot of `POST-PROCESS`. Then set the clock to the end of the run and step the frame loop by hand: 360 equal steps for
   one full turn of the finished part (14 s of page time), a screenshot after each step.
4. Make each screenshot calm (746 x 720 pixels): keep the 3D view (x 0 to 746, y 0 to 590) and, under it, the numbers part of the
   bottom bar (iteration, time, volume, compliance; x 0 to 600). The curve of the bar and the right panel (code and log) are left out.
   In the label `RUNNING · <step>`, the step name is replaced by the background, so that the label reads `RUNNING`.
5. Choose the keyframes of the forming part. Compare each pass with the pass before it (the mean grey difference of the 3D view). Choose
   15 passes so that the total change between two keyframes is about the same (the first passes change the shape much, the last passes
   little). Between two keyframes the 3D view fades from one to the next in 0.7 s, slow at both ends. The numbers bar does not fade: it
   changes at the middle of the fade. So the numbers change about once a second, not once for each pass.
6. Put the parts in order, 30 frames each second: the first pass for 1.2 s, the fades, the last pass for 1 s, a fade of 0.6 s to
   `POST-PROCESS` and 1 s on it, a hard cut to the finished part, one turn in 12 s (each frame is a real screenshot of step 3, no fade and
   no blend between them), and 0.8 s at the end.
7. Encode:

```
ffmpeg -framerate 30 -i v%05d.png -vf format=yuv420p -c:v libx264 -preset slow -crf 27 -movflags +faststart hero.mp4
ffmpeg -framerate 30 -i v%05d.png -vf "fps=10,scale=600:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" -loop 0 hero.gif
```

   The GIF keeps every third frame (10 each second); it does not blend frames.
8. Check the result for fast text and fast motion: compare each frame with the frame before it. In the clip of 2026-10-25 the numbers
   change 16 times (about 1.3 times a second while the part forms), and the mean change of the 3D view from one frame to the next is 0.92 (the
   earlier clip: 2.71; grey levels 0 to 255).

The browser needs WebGL for the 3D view. A software renderer is enough (for Chromium: `--use-angle=swiftshader`).
The `REPLAY` label of the dashboard stays in the clip: the pictures show a recording, not a new run.

## The footbridge clip

`hero.gif` and `hero.mp4` show the run of the footbridge tile of [the examples](../../examples/README.md): the region files and settings of
`examples/footbridge/footbridge.yaml` (1.1 mm voxels, 200 passes, `settings.solver: auto`, which picked `gpu-mg` on the integrated GPU of an
Intel Core Ultra 9 185H host), 2026-10-28, on main 5f2a523. The mass went from 650.9 g to 186.1 g, and the run took 2,314.9 s.
The run folder is 222 MB (173 MB of it are the 201 snapshots), too big for `recordings/`, so it is not in the repo.

The steps above, with these changes:

- The browser draws at a device scale factor of 1.5 (frames of 1920 x 1080 pixels), and the crop of step 4 is about 1.5 times larger: 1116 x 1080 pixels.
- Step 2 takes a screenshot of every second pass (101 passes from 0 to 199). Step 5 chooses 11 keyframes with the same rule.
- The dashboard of the capture was main 5f2a523 with two fixes of the region boxes and the load labels of the 3D view that were not merged yet. Main has them now
  in another form (#444 and #459).
- The GIF has no scale step: `fps=10,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle`.
- The order of step 6, with 10 fades instead of 14: 1.2 s, 7.0 s of fades, 1.0 s, 1.6 s to `POST-PROCESS` and on it, the turn in 12 s, 0.8 s.
  708 frames, 23.6 s.
- Step 8: the numbers change 12 times (about 1.3 times a second while the part forms), and the mean change of the 3D view from one frame
  to the next is 1.38 (on frames 1.5 times larger than those of the hook clip).

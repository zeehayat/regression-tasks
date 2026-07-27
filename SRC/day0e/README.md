# Day 0E Masterclass

This is a 24-segment, approximately 96-minute captioned masterclass. Parts I–III are a first sitting; Parts IV–VIII are a second sitting.

## Re-render

```bash
python3 day0e/build_day0e.py
python3 code/test_all.py
```

The build uses NumPy, Pillow, SymPy-compatible code examples, ffmpeg, and the installed offline espeak-ng female voice. Edit `SEGMENTS` and add one segment script/module to extend it. The deck is `deck/index.html`; print it to produce `deck/day0e_deck.pdf`. The existing Day 0E modal loads `day_0e_masterclass_video.html`; copy or link `video/index.html` to that page when deploying.

The deck PDF is exported with headless Chrome after rebuilding the HTML deck. The current build uses one compact reference slide per segment; the seven-slide reveal pattern is represented in the per-segment Markdown modules and can be expanded into separate reveal sections when a slide-by-slide print layout is needed. The 96-minute runtime includes deliberate exercise holds; narration is concise and the remaining time is reserved for the viewer to type and verify each exercise.

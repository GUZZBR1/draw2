# Draw — Vision

*Vision document written from Lucas Santleme's original idea, recorded on 28/10/2026.*

## The idea in one sentence

Draw is a real-time bridge between drawing on paper and professional
digital systems: the person draws on paper, the natural way, and the
system captures, interprets, and reproduces that drawing in CAD software
and simulators — giving immediate feedback while the person is still
drawing.

## Why paper?

Engineers, architects, and, in practice, most people think
first on paper. Drawing on paper has no friction: there are no menus,
tools, layers, or mouse. It is thought going straight to the stroke.

The powerful systems — AutoCAD, NVIDIA Isaac Sim / Isaac Lab, physics
and materials simulators — charge the opposite price: to use them, you
need to stop thinking about the problem and think about the tool,
drawing with the mouse, point by point.

Draw eliminates this trade-off. Paper becomes the input interface these
systems never had.

## The central loop

1. **Draw on paper** — with pencil, pen, and eraser, as always.
2. **Capture and understand** — a webcam pointed at the table captures
   each stroke. Computer vision, neural networks, and vision models
   transform the raw stroke into a structured model: a line stops being
   "ink" and becomes a wall, a beam, a part edge, or the trajectory
   of a robotic arm.
3. **Reproduce in the target system** — the model enters automatically
   into AutoCAD, a robotics simulator, or another professional system,
   as if the person were drawing inside it with the mouse — only they
   are drawing on paper.
4. **Receive feedback right away** — the simulation runs over the
   drawing while it is being made: does the resistance of the materials
   make sense? Does the shape make sense? Is there a collision, an
   overload, a stress point? The person sees the result, erases with
   the eraser, corrects it on paper — and the correction is also
   captured and applied to the model.

## The layers

The vision is grand, and it is built in layers, each one delivering
something real:

| Layer | What it does | Status |
|---|---|---|
| 1. Capture | Webcam → rectified sheet → digital ink in real time (the "digitizing table without a table") | **v0 — under construction** |
| 2. Interpretation | Raw stroke → primitives and objects with meaning, scale, and parameters | future |
| 3. Integration | Model → AutoCAD, Isaac Sim/Lab and other simulators | future |
| 4. Feedback | Simulation (materials, physics, collision) returning results in real time to the person drawing | future |

## The first barrier

Everything starts with layer 1: proving, on a real table, that an
ordinary webcam can transform what is drawn on a sheet of paper into a
faithful digital stroke, in real time — including erasing with the
eraser and continuing to draw. Without reliable capture, there is no
interpretation; without interpretation, there is no simulation.

It is this barrier that v0 tackles.

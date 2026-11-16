---
name: DynaMoE
description: Local SSD-streamed inference for Apple Silicon, presented as a dense instrument panel.
colors:
  signal-violet: "#AF52DE"
  violet-wash: "rgba(175, 82, 222, 0.18)"
  violet-faint: "rgba(175, 82, 222, 0.12)"
  violet-ghost: "rgba(175, 82, 222, 0.08)"
  violet-trace: "rgba(175, 82, 222, 0.04)"
  signal-indigo: "#5856D6"
  signal-cyan: "#32ADE6"
  status-green: "#34C759"
  status-orange: "#FF9500"
  status-red: "#FF3B30"
  action-blue: "#007AFF"
  settings-pane: "#FFFFFF"
  settings-card: "#F8F8F8"
  settings-pane-dark: "#1E1E1E"
  settings-card-dark: "#2A2A2A"
typography:
  display:
    fontFamily: "SF Pro (system), -apple-system"
    fontSize: "22px"
    fontWeight: 700
    lineHeight: 1.2
  headline:
    fontFamily: "SF Pro (system), -apple-system"
    fontSize: "19px"
    fontWeight: 700
    lineHeight: 1.25
  title:
    fontFamily: "SF Pro (system), -apple-system"
    fontSize: "16.5px"
    fontWeight: 700
    lineHeight: 1.3
  body:
    fontFamily: "SF Pro (system), -apple-system"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: "20px"
  supporting:
    fontFamily: "SF Pro (system), -apple-system"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: "17px"
  label:
    fontFamily: "SF Pro (system), -apple-system"
    fontSize: "11px"
    fontWeight: 500
    lineHeight: "14px"
  mono:
    fontFamily: "SF Mono (system monospaced)"
    fontSize: "12.5px"
    fontWeight: 400
    lineHeight: "17px"
rounded:
  composer: "18px"
  bubble: "14px"
  card: "10px"
  control: "8px"
  row: "6px"
  hairline: "4px"
  badge: "999px"
spacing:
  xxs: "2px"
  xs: "4px"
  sm: "6px"
  md: "8px"
  lg: "10px"
  xl: "14px"
  xxl: "18px"
components:
  message-user:
    backgroundColor: "{colors.violet-wash}"
    rounded: "{rounded.bubble}"
    padding: "10px 16px"
  composer:
    rounded: "{rounded.composer}"
    padding: "12px"
  code-block:
    rounded: "{rounded.control}"
    padding: "12px"
  settings-card:
    backgroundColor: "{colors.settings-card}"
    rounded: "{rounded.card}"
    padding: "6px 14px"
  icon-tile:
    rounded: "6.5px"
    size: "26px"
  status-badge:
    rounded: "{rounded.badge}"
    padding: "2px 6px"
---

# Design System: DynaMoE

## Overview

**Creative North Star: "The Telemetry Deck"**

DynaMoE looks like equipment that is running something: dense rows of small type, live numbers in monospaced figures, and a status light that changes color while the machine works. The interface is not a stage for the model; it is the panel you watch the model through. Density is the aesthetic, and restraint is how it stays readable — panels, hairlines, and system surfaces carry the structure, while one violet signal marks identity and activity.

The system is deliberately mostly stock macOS. Surfaces come from AppKit semantic colors, controls default to bordered and plain button styles, and the Settings window mirrors System Settings' inset-group pattern closely enough that a Mac user can read it without instruction. Custom work is concentrated where the product has something real to say: the transcript bubbles, the tool-execution timeline, and the streaming indicators that make off-disk inference legible.

The palette is **provisional**. Purple was an early choice rather than a brand decision, and a future rebrand will derive the palette from the DRP and DynaMoE websites. Until that lands, treat every color here as incumbent implementation truth, not as vocabulary to extend into new components.

**Key Characteristics:**
- Small type (9–13 pt) at almost every size, with a single 22 pt display moment.
- Flat system surfaces separated by 1 px translucent strokes, not shadows.
- One violet-to-indigo signal gradient, reserved for identity marks.
- State colors (green, orange, red, indigo, violet) that always mean something.
- Radii that step with containment: 18 floating, 14 bubble, 10 card, 8 control, 6 row.

## Colors

A neutral macOS canvas with one violet signal and a strict five-color status vocabulary.

### Primary
- **Signal Violet** (`signal-violet`): the product's identity and live-activity color. It appears as a 12% fill behind the assistant avatar, an 18% wash for user bubbles, an 8% pill behind progress text, a 4% trace behind blockquotes, and as the idle dot in the sidebar status pill. Its opacity tiers are tokens too (`violet-faint`, `violet-ghost`, `violet-trace`), and the gradient built from it and **Signal Indigo** is the one decorative element the system allows.
- **Signal Indigo** (`signal-indigo`): the gradient's second stop and the "running" state color in the tool timeline. It also tints the agent-tools toggle when that mode is on.
- **Signal Cyan** (`signal-cyan`): the third gradient stop on the welcome mark and the agent-working indicator, and the color of JetSpec's speculative-decoding readout.

### Secondary
- **Action Blue** (`action-blue`): reserved for the Turbo Mode toggle and for system-provided accent moments. It is not a general-purpose link color; macOS already styles links.

### Neutral
- **Window Surface** (`NSColor.windowBackgroundColor`): the transcript and sidebar ground.
- **Control Surface** (`NSColor.controlBackgroundColor`): the floating composer card, code-block header bars, and grouped panels.
- **Text Surface** (`NSColor.textBackgroundColor`): the code canvas inside a code block.
- **Settings Pane** (`settings-pane` / `settings-pane-dark`): the explicit white / near-black ground of the Settings window, chosen to match System Settings rather than inheriting the window color.
- **Settings Card** (`settings-card` / `settings-card-dark`): the faint grey group fill that replaces a border inside Settings.

### Named Rules
**The Provisional Palette Rule.** The palette is scheduled for replacement from the DRP and DynaMoE websites. Do not mint new colors from it or extend violet into a new component; use the existing tier that already means what you need, or use a system semantic color.

**The Signal-Not-Decoration Rule.** Violet identifies DynaMoE or marks that it is working. It is never a background wash for a section, a heading color, or a hover state on neutral chrome.

**The Status Vocabulary Rule.** Green means success or generating, orange means warning, attention, or Turbo Mode, red means failure, indigo means running, violet means withheld or idle. A status color always carries a state; a decorative accent never borrows one.

## Typography

**Display Font:** SF Pro (system), with `-apple-system`
**Body Font:** SF Pro (system), with `-apple-system`
**Label/Mono Font:** SF Mono via `design: .monospaced` for all numbers, timings, and identifiers.

**Character:** System type at instrument scale. The interface reads like a control panel rather than a document: measurements are monospaced, labels are small and medium-weight, and the only large text is a single greeting.

### Hierarchy
- **Display** (700, 22 pt at 1.0 zoom): the welcome greeting and the app's About mark. One per screen at most.
- **Headline** (700, 19 pt): markdown H1 inside an answer.
- **Title** (700, 16.5 pt): markdown H2 and pane titles.
- **Body** (400, 14 pt, 20 pt line height): assistant answers, user bubbles, settings row titles at 13 pt.
- **Supporting** (400, 13 pt): descriptions, status lines, and secondary labels in the transcript.
- **Label** (500–600, 11 pt; 9.5 pt uppercase bold for badges): telemetry, chips, badges, and code-block headers.
- **Mono** (400–600, 12.5 pt): code, TTFT/tok-s readouts, tool names, memory figures, keyboard hints.

### Named Rules
**The Small-Type Rule.** UI chrome lives between 9 and 13 pt. Nothing above 16.5 pt appears more than once per surface, and only as a display moment.

**The Zoom-Not-Dynamic-Type Rule.** Type scales through the app zoom (`AppZoomManager`, 0.6–2.5 in 0.15 steps, persisted), never through per-view size variants. Every custom size is written as `max(floor, base * zoomScale)` so text never falls below its legibility floor; new text must do the same.

**The Mono-Means-Measured Rule.** Any number the user might compare (tok/s, TTFT, GB, tokens, τ, durations) is monospaced. Prose numbers are not.

## Layout

A two-column workspace: a sidebar of 220–320 pt (scaled by zoom) and a transcript with a centered, width-constrained floating composer. The default window is 880 × 680 and uses content-size resizability, so the app never invents a layout for a size it was not designed at.

Vertical rhythm runs on a 2 pt grid with 4/6/8/10 pt dominating inside groups and 14/18 pt between them. Transcript rows separate by 14 pt; markdown blocks by 10 pt; settings rows by 8 pt of vertical padding inside a card. Horizontal insets cluster at 6/8/10 for chrome, 12/14 for cards and code, and 16 for bubbles.

Density is intentional and does not relax at large window sizes. The transcript widens; the type and the row heights do not.

## Elevation & Depth

The system is flat. Depth is conveyed by tonal layering — window surface behind control surface behind text surface — and by 1 px strokes at 6–20% primary opacity. Shadows exist only as a whisper under things that float above the window, and as a single violet glow on the About identity mark.

### Shadow Vocabulary
- **Floating Surface** (`0 4px 10px rgba(0, 0, 0, 0.14)`): the composer and the plan/compaction cards that sit over the transcript.
- **Transient Notice** (`0 3px 8px rgba(0, 0, 0, 0.12)`): status notices and toasts.
- **Soft Lift** (`0 4px 12px rgba(0, 0, 0, 0.08)`): secondary floating panels.
- **Whisper** (`0 4px 10px rgba(0, 0, 0, 0.06)`): the transcript's floating bottom fade layer.
- **Signal Glow** (`0 4px 8px rgba(175, 82, 222, 0.35)`): the About window's mark only.

### Named Rules
**The Hairline Rule.** Separation is a stroke, not a shadow. Inside the window, use `primary.opacity(0.06–0.2)` at 1 px; reserve every shadow for elements that leave the surface plane.

## Shapes

Corners step with containment, and every radius in the system is one of five values: 18 pt for the floating composer, 14 pt for user bubbles and counted badges, 10 pt for settings cards and toggle chips, 8 pt for code blocks, accordions, and standard controls, 6 pt for sidebar rows and inline status pills, with 4 pt on small icon buttons and true capsules for status badges.

Settings icon tiles use continuous corners (6.5 pt) to match System Settings; elsewhere the default circular curve is used. Circles are reserved for avatars and status dots. The assistant avatar is a 28 pt circle with a 12% violet fill.

Borders are always 1 px. There are no 2 px strokes, no double borders, and no outlines that are not a state.

## Components

### Buttons
- **Shape:** 6–8 pt radius for icon and row buttons; native macOS bezel for bordered buttons.
- **Primary:** `.borderedProminent` with the system accent, used sparingly for one affirmative action per pane (Load Model, Quantize, Start Server).
- **Secondary:** `.bordered` for the majority of settings actions.
- **Ghost:** `.plain` with a 6% primary fill on hover — the sidebar row and toolbar-chip idiom, used 52 times and the app's default button shape.
- **Hover / Focus:** fill appears on hover rather than a border; focus stays with the platform.

### Chips / Badges
- **Style:** uppercase 9.5 pt bold label on a 12% status tint, capsule-clipped, 2 × 6 pt padding. Optionally led by an 8 pt symbol.
- **State:** the tint is the state (indigo running, green success, red error, orange approval, violet withheld), so a chip is never neutral-colored decoration.

### Cards / Containers
- **Corner Style:** 10 pt continuous in Settings, 8 pt in the transcript.
- **Background:** `settings-card` inside Settings; `controlBackgroundColor` for transcript panels.
- **Shadow Strategy:** none at rest; Floating Surface only when the card leaves the plane.
- **Border:** none in Settings (the fill is the separation); 1 px at 6–8% primary in the transcript.
- **Internal Padding:** 6 pt vertical / 14 pt horizontal in Settings; 12 pt in transcript panels.

### Inputs / Fields
- **Style:** the composer is a native text view on `controlBackgroundColor` with an 18 pt radius and no border.
- **Focus:** the composer does not draw a focus ring; its placeholder text and the mode chips carry the state.
- **Mode chips:** attached-mode buttons (Plan Mode, Agent Tools, Turbo) sit on a 8–12% tint with a 1 px stroke at 25–40% of the same hue, and drop to `secondary.opacity(0.08)` when off.

### Navigation
The sidebar is a flat list on `windowBackgroundColor`: 8 pt horizontal insets, 2 pt row spacing, 6 pt row radius, selected rows on a 12% violet fill and hover on 6% primary. The header is one New Chat row with the same ghost-button treatment. Dividers are inset 8 pt. The bottom stack carries the loaded-model status pill (8 pt secondary fill, 6 pt radius, violet/green dot) above the Settings row.

### Signature Components
- **Assistant avatar:** 28 pt circle, 12% violet fill, `circle.hexagongrid.fill` in a violet→indigo gradient — the app's brand mark at message scale.
- **Thinking accordion:** a 7 pt-padded header bar on 70% control surface with an 8 pt radius, violet→indigo brain glyph, 12 pt state label, character count in 11 pt mono, and a 1 px 6% stroke; the expanded body is 13 pt secondary text.
- **Tool step badge:** a 6 pt status dot plus step number and tool name in 10.5 pt semibold mono, on the status tint at 8% with a 20% matching stroke and 6 pt radius.
- **Streaming pace indicator:** a tortoise/hare glyph (orange off-disk, violet in-RAM) that fades on a 0.85 s loop and appends a trailing glyph to the last streamed line.

## Do's and Don'ts

### Do:
- **Do** draw surfaces from AppKit semantic colors first; the only literal neutrals are the two Settings fills.
- **Do** write custom type as `max(floor, base * zoomScale)` so the 0.6 zoom floor never clips.
- **Do** use the status vocabulary for status and nothing else.
- **Do** keep the composer, cards, and code blocks inside the 6/8/10/14/18 radius ladder.
- **Do** prefer a 1 px stroke or a tonal fill over a shadow for separation.
- **Do** lean on stock macOS chrome — bordered buttons, menus, pickers — for new settings surfaces; the restrained direction wins ties.

### Don't:
- **Don't** introduce new hues, gradients, or violet opacity tiers; the palette is provisional and awaiting the site-derived rebrand.
- **Don't** put the violet→indigo gradient on chrome; it belongs to identity marks (avatar, welcome mark, blockquote rail, agent indicator).
- **Don't** animate longer than 0.2 s for state changes; 0.85 s is reserved for the streaming pace pulse.
- **Don't** use Dynamic Type APIs or hardcoded point sizes that ignore the zoom manager.
- **Don't** use SF Symbols larger than 30 pt outside the single display moment.

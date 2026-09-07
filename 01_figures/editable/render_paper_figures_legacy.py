#!/usr/bin/env python3
"""Render print-size QECO-ADAPT figures from existing comparison CSVs.

The six draft-referenced PNGs are regenerated at the one-column HWPX width.
No experiment is run and no metric value is altered.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
USER_500_CSV = ROOT / "results" / "comparisons" / "user_500_seed_42_ep_500_edge_3_ap_6_channel_4" / "comparison_timeseries.csv"
USER_1500_CSV = ROOT / "results" / "comparisons" / "ablation_user_1500_seed_42_ep_500_edge_3_ap_6_channel_4" / "comparison_timeseries.csv"
DEFAULT_OUTPUTS = (
    ROOT / "docs" / "topology_structure.png",
    ROOT / "docs" / "user_perspective_data_flow.png",
    ROOT / "docs" / "server_perspective_data_flow.png",
    ROOT / "results" / "comparisons" / "user_500_seed_42_ep_500_edge_3_ap_6_channel_4" / "QoE_chart.png",
    ROOT / "results" / "comparisons" / "ablation_user_1500_seed_42_ep_500_edge_3_ap_6_channel_4" / "selected4_QoE_Timeseries_0_300.png",
    ROOT / "results" / "comparisons" / "ablation_user_1500_seed_42_ep_500_edge_3_ap_6_channel_4" / "selected4_CompletionRate_Timeseries_300_500.png",
)

# The HWPX builder keeps every figure at a 5.83 in one-column width.
# These source figures therefore remain at 600 dpi and use at least 84 px
# (about 10 pt at final placement) for reader-facing chart text.  The canvas
# is tightened around its content instead of enlarging the HWPX object.
DPI = 600
WIDTH = 3540
FONT_REGULAR = Path("/opt/homebrew/Library/Homebrew/vendor/portable-ruby/4.0.6/lib/ruby/gems/4.0.0/gems/rdoc-7.0.4/lib/rdoc/generator/template/darkfish/fonts/Lato-Regular.ttf")
FONT_BOLD = FONT_REGULAR

WHITE = "#FFFFFF"
INK = "#17252A"
MUTED = "#3D4A50"
GRID = "#B8C1C8"
ARROW = "#34495E"
COLORS = {
    "tang_wong": "#C05700",
    "qeco": "#0072B2",
    "qeco_gate_only": "#A33A74",
    "qeco_adaptive_weight_fixed_gate": "#B58600",
    "qeco_adapt": "#008F67",
    "qeco_adapt_full": "#008F67",
}
LABELS = {
    "tang_wong": "Tang&Wong DQN",
    "qeco": "QECO",
    "qeco_gate_only": "QECO + gate only",
    "qeco_adaptive_weight_fixed_gate": "Adaptive weight/fixed gate",
    "qeco_adapt": "QECO-ADAPT",
    "qeco_adapt_full": "QECO-ADAPT full",
}
MARKERS = ("circle", "square", "triangle", "diamond")

# Figure-design invariants.  These apply to all regenerated paper diagrams,
# independently of legacy image layouts.
FIGURE_DESIGN_RULES = (
    "Use rectangular boxes for every labeled node.",
    "Use ASCII-safe labels and at least 84 px reader-facing text.",
    "Keep nodes disjoint and keep links out of unrelated boxes.",
    "Show static topology separately from UE execution paths.",
)


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    if not path.is_file():
        raise FileNotFoundError(f"Required rendering font is unavailable: {path}")
    return ImageFont.truetype(str(path), size)


def fit_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, max_height: int, preferred: int, *, bold: bool = False, minimum: int = 84) -> ImageFont.FreeTypeFont:
    """Return the largest consistent font that fits a diagram panel."""
    for size in range(preferred, minimum - 1, -2):
        candidate = font(size, bold=bold)
        width, height = text_size(draw, text, candidate)
        if width <= max_width and height <= max_height:
            return candidate
    raise ValueError(f"Diagram label does not fit its panel: {text!r}")


def make_canvas(height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, height), WHITE)
    return image, ImageDraw.Draw(image)


def save(image: Image.Image, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", dpi=(DPI, DPI), optimize=True)


def text_bounds(draw: ImageDraw.ImageDraw, text: str, text_font, *, spacing: int = 6) -> tuple[int, int, int, int]:
    return draw.multiline_textbbox((0, 0), text, font=text_font, spacing=spacing)


def text_size(draw: ImageDraw.ImageDraw, text: str, text_font, *, spacing: int = 6) -> tuple[int, int]:
    left, top, right, bottom = text_bounds(draw, text, text_font, spacing=spacing)
    return right - left, bottom - top


def centered_text_origin(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    text: str,
    text_font,
    *,
    spacing: int,
) -> tuple[float, float]:
    """Return an origin that centers the rendered ink bounds, not just the advance box."""
    left, top, right, bottom = text_bounds(draw, text, text_font, spacing=spacing)
    return center[0] - (left + right) / 2, center[1] - (top + bottom) / 2


def centered_text(draw: ImageDraw.ImageDraw, center: tuple[int, int], text: str, text_font, *, fill=INK, spacing=8) -> None:
    if not text.isascii():
        raise ValueError(f"Figure labels must be ASCII-safe: {text!r}")
    draw.multiline_text(
        centered_text_origin(draw, center, text, text_font, spacing=spacing),
        text,
        font=text_font,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def rounded_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, *, fill: str, text_font, radius=20) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=ARROW, width=5)
    centered_text(draw, ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), text, text_font, spacing=10)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], *, label: str | None = None) -> None:
    draw.line((start, end), fill=ARROW, width=10)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 30
    half = 14
    p1 = end
    p2 = (end[0] - length * math.cos(angle) + half * math.sin(angle), end[1] - length * math.sin(angle) - half * math.cos(angle))
    p3 = (end[0] - length * math.cos(angle) - half * math.sin(angle), end[1] - length * math.sin(angle) + half * math.cos(angle))
    draw.polygon((p1, p2, p3), fill=ARROW)
    if label:
        mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2 - 58)
        centered_text(draw, mid, label, font(78), fill=MUTED)


def group_frame(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str) -> None:
    """Outline an environment-resolved subflow without adding a title band."""
    draw.rounded_rectangle(box, radius=26, outline=ARROW, width=5)
    centered_text(draw, ((box[0] + box[2]) // 2, box[1] + 44), label, font(84), fill=MUTED)


def diamond(draw: ImageDraw.ImageDraw, center: tuple[int, int], radius: int, label: str) -> None:
    x, y = center
    points = ((x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y))
    draw.polygon(points, fill="#F4F0FF", outline=ARROW)
    draw.line((*points, points[0]), fill=ARROW, width=5, joint="curve")
    centered_text(draw, center, label, fit_font(draw, label, radius * 2 - 74, radius * 2 - 74, 88, bold=True), spacing=8)


def arrow_label(draw: ImageDraw.ImageDraw, position: tuple[int, int], label: str) -> None:
    centered_text(draw, position, label, font(84), fill=MUTED)


def feedback_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], label: str) -> None:
    """Draw a compact feedback loop below a process row."""
    bottom = max(start[1], end[1]) + 150
    draw.line((start, (start[0], bottom), (end[0], bottom), end), fill=ARROW, width=10, joint="curve")
    angle = -math.pi / 2
    length = 34
    half = 16
    p1 = end
    p2 = (end[0] - length * math.cos(angle) + half * math.sin(angle), end[1] - length * math.sin(angle) - half * math.cos(angle))
    p3 = (end[0] - length * math.cos(angle) - half * math.sin(angle), end[1] - length * math.sin(angle) + half * math.cos(angle))
    draw.polygon((p1, p2, p3), fill=ARROW)
    centered_text(draw, ((start[0] + end[0]) // 2, bottom + 54), label, font(86), fill=MUTED)


\
def graph_node(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    radius: int,
    text: str | None,
    *,
    fill: str,
    outline: str = ARROW,
    preferred: int = 108,
) -> None:
    """Draw a compact topology node with an optional readable label."""
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=7)
    if text:
        text_font = fit_font(
            draw,
            text,
            int(radius * 1.55),
            int(radius * 1.25),
            preferred,
            bold=True,
        )
        centered_text(draw, center, text, text_font, spacing=6)


\
def graph_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    fill: str,
    preferred: int = 104,
) -> None:
    """Draw a readable rectangular node for a connected graph."""
    draw.rounded_rectangle(box, radius=28, fill=fill, outline=ARROW, width=7)
    text_font = fit_font(
        draw,
        text,
        box[2] - box[0] - 80,
        box[3] - box[1] - 44,
        preferred,
        bold=True,
    )
    centered_text(draw, ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), text, text_font, spacing=8)


def graph_edge(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: str = ARROW,
    width: int = 11,
    dashed: bool = False,
) -> None:
    """Draw a directional edge; labels remain in nodes or captions."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = math.hypot(dx, dy)
    if distance == 0:
        return
    ux, uy = dx / distance, dy / distance
    if dashed:
        dash, gap = 34, 24
        offset = 0.0
        while offset < distance - 36:
            segment_end = min(offset + dash, distance - 36)
            draw.line(
                (
                    (start[0] + ux * offset, start[1] + uy * offset),
                    (start[0] + ux * segment_end, start[1] + uy * segment_end),
                ),
                fill=color,
                width=width,
            )
            offset += dash + gap
    else:
        draw.line((start, end), fill=color, width=width)
    head = 34
    half = 16
    p1 = end
    p2 = (end[0] - head * ux + half * uy, end[1] - head * uy - half * ux)
    p3 = (end[0] - head * ux - half * uy, end[1] - head * uy + half * ux)
    draw.polygon((p1, p2, p3), fill=color)


\
\
def graph_compact_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    fill: str = WHITE,
) -> None:
    """Draw a small, still-reader-legible rectangular leaf node."""
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=ARROW, width=6)
    text_font = fit_font(
        draw,
        text,
        box[2] - box[0] - 28,
        box[3] - box[1] - 32,
        90,
        bold=True,
    )
    centered_text(draw, ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), text, text_font, spacing=4)


def graph_polyline(
    draw: ImageDraw.ImageDraw,
    points: tuple[tuple[int, int], ...],
    *,
    color: str = ARROW,
    width: int = 10,
) -> None:
    """Draw a square-elbowed structural link without an arrowhead."""
    if len(points) >= 2:
        draw.line(points, fill=color, width=width)


def graph_link(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: str = ARROW,
    width: int = 10,
    dashed: bool = False,
) -> None:
    """Draw an undirected structural mapping link without an arrowhead."""
    if not dashed:
        draw.line((start, end), fill=color, width=width)
        return
    dx, dy = end[0] - start[0], end[1] - start[1]
    distance = math.hypot(dx, dy)
    if distance == 0:
        return
    ux, uy = dx / distance, dy / distance
    offset = 0.0
    while offset < distance:
        segment_end = min(offset + 34, distance)
        draw.line(
            ((start[0] + ux * offset, start[1] + uy * offset), (start[0] + ux * segment_end, start[1] + uy * segment_end)),
            fill=color,
            width=width,
        )
        offset += 58


def graph_route(
    draw: ImageDraw.ImageDraw,
    points: tuple[tuple[int, int], ...],
    *,
    color: str = ARROW,
    width: int = 11,
) -> None:
    """Draw a routed directional link with an arrowhead only at its destination."""
    if len(points) < 2:
        return
    draw.line(points, fill=color, width=width, joint="curve")
    start, end = points[-2], points[-1]
    dx, dy = end[0] - start[0], end[1] - start[1]
    distance = math.hypot(dx, dy)
    if distance == 0:
        return
    ux, uy = dx / distance, dy / distance
    head, half = 34, 16
    p1 = end
    p2 = (end[0] - head * ux + half * uy, end[1] - head * uy - half * ux)
    p3 = (end[0] - head * ux - half * uy, end[1] - head * uy + half * ux)
    draw.polygon((p1, p2, p3), fill=color)


def validate_box_graph(
    name: str,
    boxes: dict[str, tuple[int, int, int, int]],
    routes: tuple[tuple[tuple[tuple[int, int], ...], tuple[str, ...]], ...],
    *,
    min_gap: int = 22,
) -> None:
    """Reject overlapping nodes and any route entering an unrelated node."""
    items = tuple(boxes.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1:]:
            overlaps = not (
                left[2] + min_gap <= right[0]
                or right[2] + min_gap <= left[0]
                or left[3] + min_gap <= right[1]
                or right[3] + min_gap <= left[1]
            )
            if overlaps:
                raise ValueError(f"{name}: node overlap or insufficient gap: {left_name}, {right_name}")
    for points, endpoints in routes:
        for start, end in zip(points, points[1:]):
            for sample in range(1, 100):
                ratio = sample / 100
                x = start[0] + (end[0] - start[0]) * ratio
                y = start[1] + (end[1] - start[1]) * ratio
                for node_name, box in items:
                    if node_name in endpoints:
                        continue
                    if box[0] - 8 < x < box[2] + 8 and box[1] - 8 < y < box[3] + 8:
                        raise ValueError(f"{name}: route intersects unrelated node {node_name}")


def graph_legend(draw: ImageDraw.ImageDraw, y: int) -> None:
    """Use line color, not additional prose, to distinguish the three paths."""
    entries = (("task path", "#0072B2", False), ("result path", "#B75D2E", False), ("local path", "#7A858C", True))
    x = 120
    for label, color, dashed in entries:
        graph_edge(draw, (x, y), (x + 150, y), color=color, width=10, dashed=dashed)
        centered_text(draw, (x + 310, y), label, font(84), fill=MUTED)
        x += 1080


def system_topology(output: Path) -> None:
    """Show the resource hierarchy and the UE's candidate execution choices."""
    image, draw = make_canvas(1300)
    ue_box = (1460, 70, 2080, 250)
    local_box = (2340, 70, 2960, 250)
    edge_boxes = (
        (330, 360, 890, 560, "Edge 1"),
        (1490, 360, 2050, 560, "Edge 2"),
        (2650, 360, 3210, 560, "Edge 3"),
    )
    ap_boxes = (
        (180, 680, 510, 850, "AP 1"), (710, 680, 1040, 850, "AP 2"),
        (1340, 680, 1670, 850, "AP 3"), (1870, 680, 2200, 850, "AP 4"),
        (2500, 680, 2830, 850, "AP 5"), (3030, 680, 3360, 850, "AP 6"),
    )
    channel_groups = []
    for left, _, right, _, _ in ap_boxes:
        center = (left + right) // 2
        channel_groups.append((
            (center - 175, 970, center - 15, 1090, "C1"),
            (center + 15, 970, center + 175, 1090, "C2"),
            (center - 175, 1120, center - 15, 1240, "C3"),
            (center + 15, 1120, center + 175, 1240, "C4"),
        ))
    boxes = {f"edge{index}": item[:4] for index, item in enumerate(edge_boxes, 1)}
    boxes.update({f"ap{index}": item[:4] for index, item in enumerate(ap_boxes, 1)})
    for ap_index, leaves in enumerate(channel_groups, 1):
        for leaf_index, leaf in enumerate(leaves, 1):
            boxes[f"channel{ap_index}_{leaf_index}"] = leaf[:4]
    boxes.update({"ue": ue_box, "local": local_box})

    edge_ap_routes = []
    channel_routes = []
    for index, ap in enumerate(ap_boxes):
        ap_name = f"ap{index + 1}"
        edge = edge_boxes[index // 2]
        edge_center = (edge[0] + edge[2]) // 2
        ap_center = (ap[0] + ap[2]) // 2
        edge_ap_routes.append((
            ((edge_center, edge[3]), (edge_center, 615), (ap_center, 615), (ap_center, ap[1])),
            (f"edge{index // 2 + 1}", ap_name),
        ))
        trunk = (ap_center, 910)
        bus = (ap_center, 1105)
        channel_routes.append((((ap_center, ap[3]), trunk), (ap_name,)))
        channel_routes.append(((trunk, bus), (ap_name,)))
        for leaf_index, leaf in enumerate(channel_groups[index]):
            leaf_name = f"channel{index + 1}_{leaf_index + 1}"
            leaf_center = ((leaf[0] + leaf[2]) // 2, leaf[1])
            branch = trunk if leaf_index < 2 else bus
            channel_routes.append((
                (branch, (leaf_center[0], branch[1]), leaf_center),
                (ap_name, leaf_name),
            ))

    ue_center_x = (ue_box[0] + ue_box[2]) // 2
    ue_edge_routes = tuple(
        (
            ((ue_center_x, ue_box[3]), ((edge[0] + edge[2]) // 2, edge[1])),
            ("ue", f"edge{index}"),
        )
        for index, edge in enumerate(edge_boxes, 1)
    )
    local_route = (
        (ue_box[2], (ue_box[1] + ue_box[3]) // 2),
        (local_box[0], (local_box[1] + local_box[3]) // 2),
    )
    routes = tuple(edge_ap_routes + channel_routes) + ue_edge_routes + ((local_route, ("ue", "local")),)
    validate_box_graph("Figure 1", boxes, routes, min_gap=10)

    # Figure captions carry the title; the panel itself is reserved for topology information.
    for points, _ in edge_ap_routes:
        graph_polyline(draw, points, color="#0072B2", width=8)
    for points, _ in channel_routes:
        graph_polyline(draw, points, color="#8C6D1F", width=8)
    for points, _ in ue_edge_routes:
        graph_link(draw, *points, color="#6B5B95", width=8, dashed=True)
    graph_link(draw, *local_route, color="#7A858C", dashed=True)

    for left, top, right, bottom, label in edge_boxes:
        graph_box(draw, (left, top, right, bottom), label, fill="#E8F1F8")
    for left, top, right, bottom, label in ap_boxes:
        graph_box(draw, (left, top, right, bottom), label, fill="#E4F3EA", preferred=98)
    for leaves in channel_groups:
        for leaf in leaves:
            graph_compact_box(draw, leaf[:4], leaf[4], fill="#FFF3D9")
    graph_box(draw, ue_box, "UE agents", fill="#F4F0FF", preferred=102)
    graph_box(draw, local_box, "Local compute", fill="#F1F3F5", preferred=98)
    save(image, output)

def ue_flow(output: Path) -> None:
    """Show the UE decision path; validation and drawing share one route definition."""
    image, draw = make_canvas(950)
    blue, brown, gray = "#0072B2", "#B75D2E", "#7A858C"
    state_box = (100, 100, 560, 310)
    policy_box = (700, 100, 1160, 310)
    action_box = (1300, 100, 1760, 310)
    cluster_box = (1900, 100, 2380, 310)
    channel_box = (2520, 100, 2940, 310)
    queue_box = (3070, 100, 3460, 310)
    local_box = (1300, 510, 1760, 730)
    reward_box = (2240, 580, 2820, 820)
    boxes = {
        "state": state_box, "policy": policy_box, "action": action_box,
        "cluster": cluster_box, "channel": channel_box, "queue": queue_box,
        "local": local_box, "reward": reward_box,
    }
    state_to_policy = ((state_box[2], 205), (policy_box[0], 205))
    policy_to_action = ((policy_box[2], 205), (action_box[0], 205))
    action_to_cluster = ((action_box[2], 205), (cluster_box[0], 205))
    cluster_to_channel = ((cluster_box[2], 205), (channel_box[0], 205))
    channel_to_queue = ((channel_box[2], 205), (queue_box[0], 205))
    action_to_local = (((action_box[0] + action_box[2]) // 2, action_box[3]), ((local_box[0] + local_box[2]) // 2, local_box[1]))
    # Return remote execution feedback through an exclusive lane that enters
    # the centre of Reward's top edge, rather than its side.
    queue_to_reward = (
        ((queue_box[0] + queue_box[2]) // 2, queue_box[3]),
        ((queue_box[0] + queue_box[2]) // 2, 440),
        ((reward_box[0] + reward_box[2]) // 2, 440),
        ((reward_box[0] + reward_box[2]) // 2, reward_box[1]),
    )
    local_to_reward = ((local_box[2], 630), (reward_box[0], 700))
    reward_to_state = ((reward_box[0], reward_box[3]), (330, reward_box[3]), (330, state_box[3]))
    routes = (
        (state_to_policy, ("state", "policy")),
        (policy_to_action, ("policy", "action")),
        (action_to_cluster, ("action", "cluster")),
        (cluster_to_channel, ("cluster", "channel")),
        (channel_to_queue, ("channel", "queue")),
        (action_to_local, ("action", "local")),
        (queue_to_reward, ("queue", "reward")),
        (local_to_reward, ("local", "reward")),
        (reward_to_state, ("reward", "state")),
    )
    validate_box_graph("Figure 2", boxes, routes)

    for box, label, fill in (
        (state_box, "State", "#E8F1F8"),
        (policy_box, "Policy", "#E4F3EA"),
        (action_box, "Action", "#FFF3D9"),
        (cluster_box, "AP cluster", "#FFF3D9"),
        (channel_box, "Channel", "#FCE7E6"),
        (queue_box, "Queue", "#E9EDF8"),
        (local_box, "Local", "#F1F3F5"),
        (reward_box, "Reward", "#EFF6F2"),
    ):
        graph_box(draw, box, label, fill=fill, preferred=100)
    graph_edge(draw, *state_to_policy, width=13)
    graph_edge(draw, *policy_to_action, width=13)
    graph_edge(draw, *action_to_cluster, color=blue, width=14)
    graph_edge(draw, *cluster_to_channel, color=blue, width=14)
    graph_edge(draw, *channel_to_queue, color=blue, width=14)
    graph_edge(draw, *action_to_local, color=gray, width=11, dashed=True)
    graph_route(draw, queue_to_reward, color=brown, width=11)
    graph_edge(draw, *local_to_reward, color=brown, width=11)
    graph_route(draw, reward_to_state, color=brown, width=11)
    save(image, output)

def edge_flow(output: Path) -> None:
    """Show edge processing without any node-overlap or route-crossing ambiguity."""
    image, draw = make_canvas(1180)
    blue, brown = "#0072B2", "#B75D2E"
    ingress_box = (100, 300, 550, 510)
    queue_box = (700, 300, 1150, 510)
    compute_box = (1300, 300, 1810, 510)
    check_box = (1960, 300, 2380, 510)
    complete_box = (2900, 150, 3400, 350)
    drop_box = (2900, 610, 3400, 810)
    reward_box = (1940, 850, 2580, 1090)
    ue_box = (650, 880, 1110, 1080)
    boxes = {
        "ingress": ingress_box, "queue": queue_box, "compute": compute_box,
        "check": check_box, "complete": complete_box, "drop": drop_box,
        "reward": reward_box, "ue": ue_box,
    }
    result_center_y = (reward_box[1] + reward_box[3]) // 2
    collection_x = 3480
    complete_to_merge = (
        (complete_box[2], (complete_box[1] + complete_box[3]) // 2),
        (collection_x, (complete_box[1] + complete_box[3]) // 2),
        (collection_x, result_center_y),
    )
    drop_to_merge = (
        (drop_box[2], (drop_box[1] + drop_box[3]) // 2),
        (collection_x, (drop_box[1] + drop_box[3]) // 2),
    )
    merged_result_to_reward = (
        (collection_x, result_center_y),
        (reward_box[2], result_center_y),
    )
    routes = (
        (((ingress_box[2], 405), (queue_box[0], 405)), ("ingress", "queue")),
        (((queue_box[2], 405), (compute_box[0], 405)), ("queue", "compute")),
        (((compute_box[2], 405), (check_box[0], 405)), ("compute", "check")),
        (
            ((check_box[2], 350), (2580, 350), (2580, 250), (complete_box[0], 250)),
            ("check", "complete"),
        ),
        (
            ((check_box[2], 460), (2580, 460), (2580, 710), (drop_box[0], 710)),
            ("check", "drop"),
        ),
        (complete_to_merge, ("complete",)),
        (drop_to_merge, ("drop",)),
        (merged_result_to_reward, ("reward",)),
        (((reward_box[0], 970), (ue_box[2], 970)), ("reward", "ue")),
    )
    validate_box_graph("Figure 3", boxes, routes)

    for box, label, fill in (
        (ingress_box, "AP CH", "#E8F1F8"),
        (queue_box, "Queue", "#E4F3EA"),
        (compute_box, "Compute", "#FFF3D9"),
        (check_box, "Check", "#F4F0FF"),
        (complete_box, "Complete", "#E4F3EA"),
        (drop_box, "Drop", "#FCE7E6"),
        (reward_box, "Reward", "#E9EDF8"),
        (ue_box, "UE agent", "#EFF6F2"),
    ):
        graph_box(draw, box, label, fill=fill, preferred=100)
    graph_edge(draw, (ingress_box[2], 405), (queue_box[0], 405), color=blue, width=14)
    graph_edge(draw, (queue_box[2], 405), (compute_box[0], 405), color=blue, width=14)
    graph_edge(draw, (compute_box[2], 405), (check_box[0], 405), color=blue, width=14)
    graph_route(
        draw,
        ((check_box[2], 350), (2580, 350), (2580, 250), (complete_box[0], 250)),
        color="#008F67",
        width=13,
    )
    graph_route(
        draw,
        ((check_box[2], 460), (2580, 460), (2580, 710), (drop_box[0], 710)),
        color="#C05700",
        width=13,
    )
    # Complete and Drop share one result-collection lane. Both originate at
    # the right-centre of their outcome boxes; only the common leg carries the
    # arrowhead into the right-centre of Reward.
    graph_polyline(draw, complete_to_merge, color=brown, width=11)
    graph_polyline(draw, drop_to_merge, color=brown, width=11)
    graph_route(draw, merged_result_to_reward, color=brown, width=11)
    graph_edge(draw, (reward_box[0], 970), (ue_box[2], 970), color=brown, width=11)
    save(image, output)

def load_csv(path: Path) -> tuple[list[int], dict[str, dict[str, list[float]]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing comparison CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Comparison CSV has no data rows: {path}")
    metrics = ("qoe", "delay", "energy", "drop", "completion_rate")
    algorithms: list[str] = []
    for field in rows[0]:
        if field == "step":
            continue
        metric = next((candidate for candidate in metrics if field.endswith(f"_{candidate}")), None)
        if metric is None:
            raise ValueError(f"Unexpected column: {field}")
        algorithm = field[: -(len(metric) + 1)]
        if algorithm not in algorithms:
            algorithms.append(algorithm)
    data = {algorithm: {metric: [] for metric in metrics} for algorithm in algorithms}
    for row in rows:
        for algorithm in algorithms:
            for metric in metrics:
                data[algorithm][metric].append(float(row[f"{algorithm}_{metric}"]))
    return [int(row["step"]) for row in rows], data


def moving_average(values: list[float], window: int = 25) -> list[float]:
    total = 0.0
    result = []
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        result.append(total / min(index + 1, window))
    return result


def nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    if math.isclose(low, high):
        return [low]
    rough = (high - low) / max(count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(rough))
    multiple = min((1, 2, 2.5, 5, 10), key=lambda x: abs(rough / magnitude - x))
    step = multiple * magnitude
    start = math.floor(low / step) * step
    end = math.ceil(high / step) * step
    ticks = []
    value = start
    while value <= end + step * 0.001:
        ticks.append(value)
        value += step
    return ticks


def format_y_tick(tick: float, *, percent: bool) -> str:
    if percent:
        return f"{tick:.0f}%"
    return f"{tick:.1f}"


def chart_frame(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    start: int,
    stop: int,
    values: list[float],
    percent: bool = False,
    hidden_y_ticks: tuple[float, ...] = (),
    min_y_padding: float | None = None,
) -> tuple[int, int, int, int, float, float]:
    """Draw a compact chart frame; metric names remain in the figure caption."""
    left, top, right, bottom = 300, 80, 3370, 1660
    minimum, maximum = min(values), max(values)
    base_padding = 2.0 if percent else 0.2
    if min_y_padding is not None:
        if min_y_padding <= 0:
            raise ValueError(f"min_y_padding must be positive, got {min_y_padding}")
        base_padding = min_y_padding
    padding = max((maximum - minimum) * 0.08, base_padding)
    low, high = minimum - padding, maximum + padding
    if percent:
        low, high = max(0, low), min(100, high)

    tick_font = font(84)
    axis_font = font(96)
    draw.line((left, top, left, bottom), fill=INK, width=6)
    draw.line((left, bottom, right, bottom), fill=INK, width=6)
    for tick in nice_ticks(low, high):
        # Do not render a rounded tick just outside the selected data window.
        # This keeps zoomed plots from adding a stray gridline beyond the axis.
        if tick < low - 1e-9 or tick > high + 1e-9:
            continue
        if any(math.isclose(tick, hidden, abs_tol=1e-9) for hidden in hidden_y_ticks):
            continue
        y = bottom - (tick - low) / (high - low) * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=3)
        label = format_y_tick(tick, percent=percent)
        width, height = text_size(draw, label, tick_font)
        draw.text((left - width - 26, y - height / 2), label, font=tick_font, fill=INK)

    if stop - start > 220:
        x_ticks = [start, 50, 100, 150, 200, 250, stop]
    else:
        x_ticks = [start, 350, 400, 450, stop]
    x_ticks = sorted({tick for tick in x_ticks if start <= tick <= stop})
    for tick in x_ticks:
        x = left + (tick - start) / (stop - start) * (right - left)
        draw.line((x, bottom, x, bottom + 18), fill=INK, width=5)
        label = str(tick)
        width, _ = text_size(draw, label, tick_font)
        draw.text((x - width / 2, bottom + 28), label, font=tick_font, fill=INK)
    centered_text(draw, ((left + right) // 2, 1840), "Episode", axis_font, fill=INK)
    return left, top, right, bottom, low, high

def faded_color(color: str, *, opacity: float) -> str:
    """Return a low-contrast version of a policy color over a white background."""
    if not 0.0 <= opacity <= 1.0:
        raise ValueError(f"Opacity must be within [0, 1], got {opacity}")
    channels = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
    blended = tuple(round(255 * (1.0 - opacity) + channel * opacity) for channel in channels)
    return "#" + "".join(f"{channel:02X}" for channel in blended)


def draw_marker(draw: ImageDraw.ImageDraw, x: int, y: int, color: str, marker: str) -> None:
    r = 17
    if marker == "circle":
        draw.ellipse((x - r, y - r, x + r, y + r), fill=WHITE, outline=color, width=6)
    elif marker == "square":
        draw.rectangle((x - r, y - r, x + r, y + r), fill=WHITE, outline=color, width=6)
    elif marker == "triangle":
        points = ((x, y - r - 2), (x - r - 2, y + r), (x + r + 2, y + r))
        draw.polygon(points, fill=WHITE)
        draw.line((*points, points[0]), fill=color, width=6, joint="curve")
    else:
        points = ((x, y - r - 2), (x - r - 2, y), (x, y + r + 2), (x + r + 2, y))
        draw.polygon(points, fill=WHITE)
        draw.line((*points, points[0]), fill=color, width=6, joint="curve")

def draw_legend(image: Image.Image, entries: list[tuple[str, str, str]], *, bottom: int, right: int) -> None:
    """Place the policy key at the lower-right of the chart data panel."""
    panel = (right - 1420, bottom - 444, right - 34, bottom - 34)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(panel, radius=20, fill=(255, 255, 255, 224), outline=(52, 73, 94, 220), width=4)
    image.paste(overlay, (0, 0), overlay)
    draw = ImageDraw.Draw(image)
    legend_font = font(84)
    for index, (label, color, marker) in enumerate(entries):
        x = panel[0] + 42
        y = panel[1] + 36 + index * 91
        draw.line((x, y + 35, x + 120, y + 35), fill=color, width=12)
        draw_marker(draw, x + 60, y + 35, color, marker)
        draw.text((x + 160, y), label, font=legend_font, fill=INK)

def timeseries(
    output: Path,
    csv_path: Path,
    metric: str,
    start: int,
    stop: int,
    *,
    hidden_y_ticks: tuple[float, ...] = (),
    min_y_padding: float | None = None,
) -> None:
    steps, data = load_csv(csv_path)
    algorithms = ("qeco", "qeco_gate_only", "qeco_adaptive_weight_fixed_gate", "qeco_adapt_full")
    if any(item not in data for item in algorithms):
        raise ValueError(f"The comparison CSV lacks the four ablation algorithms: {csv_path}")
    chosen = [(step, index) for index, step in enumerate(steps) if start <= step <= stop]
    x_values = [step for step, _ in chosen]
    if not x_values:
        raise ValueError(f"No episode data in {start}-{stop}: {csv_path}")

    raw_series: dict[str, list[float]] = {}
    smoothed_series: dict[str, list[float]] = {}
    for algorithm in algorithms:
        values = [data[algorithm][metric][row_index] for _, row_index in chosen]
        if metric == "completion_rate":
            values = [100 * value for value in values]
        raw_series[algorithm] = values
        smoothed_series[algorithm] = moving_average(values)

    image, draw = make_canvas(1990)
    # Keep all episode measurements in the plotted range; otherwise faint raw
    # traces could be clipped at the panel boundary while their averages remain visible.
    all_values = [
        value
        for algorithm in algorithms
        for values in (raw_series[algorithm], smoothed_series[algorithm])
        for value in values
    ]
    left, top, right, bottom, low, high = chart_frame(
        image,
        draw,
        start=start,
        stop=stop,
        values=all_values,
        percent=metric == "completion_rate",
        hidden_y_ticks=hidden_y_ticks,
        min_y_padding=min_y_padding,
    )
    def points_for(values: list[float]) -> list[tuple[int, int]]:
        return [
            (
                round(left + (step - start) / (stop - start) * (right - left)),
                round(bottom - (value - low) / (high - low) * (bottom - top)),
            )
            for step, value in zip(x_values, values)
        ]

    # The unsmoothed episode measurements stay visible as quiet context behind
    # the moving-average trends, without adding a second legend entry per policy.
    for algorithm in algorithms:
        draw.line(
            points_for(raw_series[algorithm]),
            fill=faded_color(COLORS[algorithm], opacity=0.28),
            width=5,
            joint="curve",
        )
    for index, algorithm in enumerate(algorithms):
        points = points_for(smoothed_series[algorithm])
        draw.line(points, fill=COLORS[algorithm], width=14, joint="curve")
        marker_positions = list(range(0, len(points), max(1, len(points) // 6)))
        if marker_positions[-1] != len(points) - 1:
            marker_positions.append(len(points) - 1)
        for position in marker_positions:
            draw_marker(draw, points[position][0], points[position][1], COLORS[algorithm], MARKERS[index])
    draw_legend(image, [(LABELS[a], COLORS[a], MARKERS[index]) for index, a in enumerate(algorithms)], bottom=bottom, right=right)
    save(image, output)

def qoe_overall_bar(output: Path, csv_path: Path) -> None:
    _, data = load_csv(csv_path)
    algorithms = ("tang_wong", "qeco", "qeco_adapt")
    values = [mean(data[algorithm]["qoe"]) for algorithm in algorithms]
    image, draw = make_canvas(1420)
    label_font = font(84)
    tick_font = font(84)
    value_font = font(92, bold=True)
    left, top, right, bottom = 300, 95, 3390, 1280
    maximum = max(values) * 1.20
    for tick in nice_ticks(0, maximum):
        x = left + tick / maximum * (right - left)
        draw.line((x, top, x, bottom), fill=GRID, width=3)
        label = f"{tick:.0f}"
        width, _ = text_size(draw, label, tick_font)
        draw.text((x - width / 2, bottom + 28), label, font=tick_font, fill=INK)
    draw.line((left, top, left, bottom), fill=INK, width=6)
    draw.line((left, bottom, right, bottom), fill=INK, width=6)
    bar_height = 205
    for index, (algorithm, value) in enumerate(zip(algorithms, values)):
        y = 160 + index * 330
        width = int(value / maximum * (right - left))
        draw.rounded_rectangle((left, y, left + width, y + bar_height), radius=20, fill=COLORS[algorithm])
        label = LABELS[algorithm]
        _, label_height = text_size(draw, label, label_font)
        draw.text((left + 42, y + (bar_height - label_height) / 2), label, font=label_font, fill=WHITE)
        value_label = f"{value:.2f}"
        draw.text((left + width + 30, y + 37), value_label, font=value_font, fill=INK)
    save(image, output)

def render_all(output_dir: Path | None = None) -> list[Path]:
    if output_dir is None:
        outputs = list(DEFAULT_OUTPUTS)
    else:
        outputs = [
            output_dir / "topology_structure.png",
            output_dir / "user_perspective_data_flow.png",
            output_dir / "server_perspective_data_flow.png",
            output_dir / "QoE_chart.png",
            output_dir / "selected4_QoE_Timeseries_0_300.png",
            output_dir / "selected4_CompletionRate_Timeseries_300_500.png",
        ]
    system_topology(outputs[0])
    ue_flow(outputs[1])
    edge_flow(outputs[2])
    qoe_overall_bar(outputs[3], USER_500_CSV)
    timeseries(
        outputs[4], USER_1500_CSV, "qoe", 1, 300, hidden_y_ticks=(-15.0, 10.0)
    )
    timeseries(
        outputs[5],
        USER_1500_CSV,
        "completion_rate",
        301,
        500,
        hidden_y_ticks=(42.0, 50.0),
        min_y_padding=0.5,
    )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Render print-size QECO-ADAPT paper figures from existing comparison CSVs.")
    parser.add_argument("--output-dir", type=Path, help="Optional review directory; default overwrites the six draft-referenced PNGs.")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve() if args.output_dir else None
    for output in render_all(output_dir):
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

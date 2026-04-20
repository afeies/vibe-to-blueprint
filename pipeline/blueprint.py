"""Render a labelled top-down floor-plan blueprint from a parsed schema."""
import io
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Arc


# Rough real-world dimensions (metres) and colours for common furniture.
FURNITURE_SPECS = {
    "sofa":           {"w": 2.2, "h": 0.9, "color": "#8B7355", "label": "Sofa"},
    "couch":          {"w": 2.2, "h": 0.9, "color": "#8B7355", "label": "Couch"},
    "armchair":       {"w": 0.9, "h": 0.9, "color": "#A0826D", "label": "Armchair"},
    "chair":          {"w": 0.5, "h": 0.5, "color": "#A0826D", "label": "Chair"},
    "bed":            {"w": 1.6, "h": 2.0, "color": "#D4C4B0", "label": "Bed"},
    "nightstand":     {"w": 0.5, "h": 0.4, "color": "#B08968", "label": "Night"},
    "coffee table":   {"w": 1.0, "h": 0.6, "color": "#6B4423", "label": "Coffee"},
    "dining table":   {"w": 1.6, "h": 0.9, "color": "#6B4423", "label": "Dining"},
    "table":          {"w": 1.2, "h": 0.8, "color": "#6B4423", "label": "Table"},
    "desk":           {"w": 1.2, "h": 0.6, "color": "#6B4423", "label": "Desk"},
    "bookshelf":      {"w": 1.0, "h": 0.4, "color": "#5C3317", "label": "Books"},
    "shelf":          {"w": 0.8, "h": 0.4, "color": "#5C3317", "label": "Shelf"},
    "lamp":           {"w": 0.4, "h": 0.4, "color": "#F5DEB3", "label": "Lamp"},
    "floor lamp":     {"w": 0.4, "h": 0.4, "color": "#F5DEB3", "label": "Lamp"},
    "plant":          {"w": 0.5, "h": 0.5, "color": "#556B2F", "label": "Plant"},
    "rug":            {"w": 2.4, "h": 1.6, "color": "#E8DCC0", "label": "Rug"},
    "tv":             {"w": 1.2, "h": 0.2, "color": "#1C1C1C", "label": "TV"},
    "dresser":        {"w": 1.4, "h": 0.5, "color": "#6B4423", "label": "Dresser"},
    "wardrobe":       {"w": 1.6, "h": 0.6, "color": "#5C3317", "label": "Wardrobe"},
    "fireplace":      {"w": 1.4, "h": 0.4, "color": "#4A2C2A", "label": "Fire"},
    "kitchen island": {"w": 2.0, "h": 0.9, "color": "#8B7355", "label": "Island"},
    "counter":        {"w": 2.4, "h": 0.6, "color": "#A0826D", "label": "Counter"},
}


def _match_furniture(item: str) -> dict:
    key = item.lower().strip()
    if key in FURNITURE_SPECS:
        return FURNITURE_SPECS[key]
    # substring match
    for k, spec in FURNITURE_SPECS.items():
        if k in key or key in k:
            return spec
    return {"w": 0.8, "h": 0.8, "color": "#B0B0B0", "label": item[:6].title()}


def _place_along_walls(items: list, room_w: float, room_h: float, margin: float = 0.4) -> list:
    """Greedy placement: walk the four walls in order, placing each item along them."""
    placements = []
    # largest first for visual balance
    specs = [(it, _match_furniture(it)) for it in items]
    specs.sort(key=lambda x: -(x[1]["w"] * x[1]["h"]))

    # track walking cursors along each wall
    bottom_x = margin
    top_x = margin
    left_y = margin + 0.5   # leave clearance for door arc
    right_y = margin
    wall_order = ["bottom", "right", "top", "left"]

    for i, (name, spec) in enumerate(specs):
        wall = wall_order[i % 4]
        w, h = spec["w"], spec["h"]

        if wall == "bottom":
            x, y = bottom_x, margin
            bottom_x += w + 0.25
        elif wall == "top":
            x, y = top_x, room_h - margin - h
            top_x += w + 0.25
        elif wall == "left":
            x, y = margin, left_y
            left_y += h + 0.25
        else:  # right
            x, y = room_w - margin - w, right_y
            right_y += h + 0.25

        # Skip if it no longer fits
        if x + w > room_w - margin or y + h > room_h - margin or x < margin or y < margin:
            continue

        placements.append({
            "label": spec["label"], "color": spec["color"],
            "x": x, "y": y, "w": w, "h": h,
        })

    return placements


def render_blueprint(schema: dict) -> Image.Image:
    """Return a PIL floor-plan image derived from a parsed schema."""
    dims = schema.get("dimensions") or {}
    room_w = float(dims.get("width_m") or 5.0)
    room_h = float(dims.get("length_m") or 4.0)

    rooms = schema.get("rooms") or ["Living Room"]
    room_label = rooms[0].title() if rooms else "Room"

    furniture = schema.get("furniture") or []

    fig = Figure(figsize=(8, 8), dpi=100, facecolor="#FAFAF7")
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.6, room_w + 0.6)
    ax.set_ylim(-0.6, room_h + 0.6)
    ax.set_aspect("equal")
    ax.set_facecolor("#FAFAF7")

    wall_color = "#2C2C2C"

    # floor
    floor = Rectangle((0, 0), room_w, room_h, facecolor="#FFFFFF", edgecolor="none", zorder=1)
    ax.add_patch(floor)
    # walls
    walls = Rectangle((0, 0), room_w, room_h, facecolor="none", edgecolor=wall_color,
                      linewidth=6, zorder=4)
    ax.add_patch(walls)

    # door on bottom wall (arc swinging into room)
    door_x = 0.4
    door_w = 0.9
    ax.plot([door_x, door_x + door_w], [0, 0], linewidth=8, color="#FAFAF7",
            solid_capstyle="butt", zorder=5)
    door_arc = Arc((door_x, 0), door_w * 2, door_w * 2, angle=0, theta1=0, theta2=90,
                   linewidth=1.2, color=wall_color, linestyle="--", zorder=5)
    ax.add_patch(door_arc)
    ax.plot([door_x, door_x + door_w], [0, door_w], linewidth=1.2,
            color=wall_color, linestyle="--", zorder=5)

    # window on top wall
    win_x = room_w * 0.38
    win_w = min(room_w * 0.32, 1.8)
    ax.plot([win_x, win_x + win_w], [room_h, room_h], linewidth=8, color="#FAFAF7",
            solid_capstyle="butt", zorder=5)
    ax.plot([win_x, win_x + win_w], [room_h, room_h], linewidth=2.5, color="#4682B4", zorder=6)
    for t in (0.0, 0.5, 1.0):
        ax.plot([win_x + win_w * t, win_x + win_w * t], [room_h - 0.1, room_h + 0.1],
                linewidth=1.2, color="#4682B4", zorder=6)

    # furniture
    for p in _place_along_walls(furniture, room_w, room_h):
        rect = Rectangle((p["x"], p["y"]), p["w"], p["h"], facecolor=p["color"],
                         edgecolor="#2C2C2C", linewidth=1.2, alpha=0.88, zorder=3)
        ax.add_patch(rect)
        ax.text(p["x"] + p["w"] / 2, p["y"] + p["h"] / 2, p["label"],
                ha="center", va="center", fontsize=7, color="white", weight="bold", zorder=4)

    # room label
    ax.text(room_w / 2, room_h + 0.35, room_label,
            ha="center", va="bottom", fontsize=14, weight="bold", color="#2C2C2C",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFFF", edgecolor="#CCC"))

    # dimensions
    ax.annotate(f"{room_w:.1f} m", xy=(room_w / 2, -0.3), ha="center", va="top",
                fontsize=10, color="#666")
    ax.annotate(f"{room_h:.1f} m", xy=(-0.3, room_h / 2), ha="right", va="center",
                fontsize=10, color="#666", rotation=90)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#FAFAF7")
    buf.seek(0)
    return Image.open(buf).convert("RGB")

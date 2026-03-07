"""
visualize_sd_coords.py
----------------------
Visualises the (s, d) curvilinear coordinate system of an SDCorrectedImage.

All coordinate data and boundary decomposition are read directly from the
attributes cached by SDCorrectedImage._build_sd_maps() — no geometry is
recomputed here.

Four panels:
  1. Boundary decomposition: skin wall (lime) vs inferred crop edge (orange),
     with accurate s=0 / s=1 frame-edge origins marked.
  2. Depth map  d(x,y)
  3. Arc-length s(x,y), with accurate s=0 / s=1 origins marked.
  4. Composite overlay: iso-d (solid) and iso-s (dashed) contour lines.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MaxNLocator


def visualize_sd_coords(
    sd_image,
    n_iso_d: int   = 10,
    n_iso_s: int   = 16,
    cmap_d: str    = "plasma",
    cmap_s: str    = "hsv",
    figsize: tuple = (18, 10),
    save_path      = None,
    dpi: int       = 150,
):
    """
    Visualise the (s, d) curvilinear coordinate system of an SDCorrectedImage.

    _build_sd_maps() is called if not already done.  The crop-edge detection
    parameters (straightness_window, angle_tol_deg, dilation_px) are controlled
    there, not here.

    Parameters
    ----------
    sd_image  : SDCorrectedImage
    n_iso_d   : number of iso-depth contour lines (default 10)
    n_iso_s   : number of iso-arc-length contour lines (default 16)
    cmap_d    : colormap for depth map panel
    cmap_s    : colormap for arc-length map panel
    figsize   : figure size in inches
    save_path : if given, save figure to this path
    dpi       : resolution for saved figure

    Returns
    -------
    fig, axes
    """
    # ── Build maps if needed ──────────────────────────────────────────────────
    sd_image._build_sd_maps()

    img       = sd_image.pixel_array.astype(np.float64)
    mask      = sd_image._sd_mask
    d_norm    = sd_image._sd_d_norm
    s_norm    = sd_image._sd_s_norm
    boundary  = sd_image._sd_boundary    # (N, 2) full contour, (row, col)
    on_edge   = sd_image._sd_on_edge     # (N,) bool
    skin_ring = sd_image._sd_skin_ring   # (H, W) bool
    edge_ring = sd_image._sd_edge_ring   # (H, W) bool
    rows, cols = img.shape

    # ── s-coordinate true origins (image-frame endpoints) ────────────────────
    # _sd_s_origin / _sd_s_end are set by the patched _build_sd_maps and
    # represent the points snapped to the physical image border.  They may
    # differ from skin_contour[0] / skin_contour[-1] (the skin-wall transition
    # corners) when the breast does not exit the frame cleanly.
    s_origin = sd_image._sd_s_origin   # (row, col) array, s = 0
    s_end    = sd_image._sd_s_end      # (row, col) array, s = s_total

    # Skin-wall transition corners (first/last skin contour points)
    skin_contour  = boundary[~on_edge]
    edge_contour  = boundary[ on_edge]
    n_edge_pts    = on_edge.sum()
    pct_edge      = 100.0 * n_edge_pts / len(boundary)
    has_crop_edge = n_edge_pts > 0

    # ── Display arrays ────────────────────────────────────────────────────────
    d_display = np.where(mask, d_norm, np.nan)
    s_display = np.where(mask, s_norm, np.nan)

    edge_overlay = np.zeros((rows, cols, 4), dtype=np.float32)
    edge_overlay[edge_ring] = [1.0, 0.55, 0.0, 0.88]

    sr_r, sr_c = np.where(skin_ring)

    # ── Legend proxies ────────────────────────────────────────────────────────
    leg_skin     = Line2D([0], [0], color="lime",   lw=1.8,
                          label="Skin wall  (EDT seed & arc-length reference)")
    leg_edge     = Line2D([0], [0], color="orange", lw=1.8, linestyle="--",
                          label=f"Inferred crop edge  "
                                f"({n_edge_pts} pts, {pct_edge:.1f}% of contour)")
    leg_s0_frame = Line2D([0], [0], marker="*", color="w", lw=0,
                          markerfacecolor="red",    markersize=11,
                          label="s = 0  (frame-edge origin)")
    leg_s1_frame = Line2D([0], [0], marker="*", color="w", lw=0,
                          markerfacecolor="deepskyblue", markersize=11,
                          label="s = 1  (frame-edge endpoint)")
    leg_s0_skin  = Line2D([0], [0], marker="o", color="w", lw=0,
                          markerfacecolor="red",    markersize=7,  alpha=0.55,
                          label="s = 0  skin transition corner")
    leg_s1_skin  = Line2D([0], [0], marker="o", color="w", lw=0,
                          markerfacecolor="deepskyblue", markersize=7, alpha=0.55,
                          label="s = 1  skin transition corner")

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        "Curvilinear (s, d) Coordinate System\n"
        "Crop edges inferred from contour geometry  "
        f"({n_edge_pts} pts flagged, {pct_edge:.1f}% of contour)",
        fontsize=13, fontweight="bold",
    )
    ax_img, ax_d, ax_s, ax_ov = axes.ravel()

    vmin, vmax = np.nanpercentile(img[mask], [1, 99])
    row_grid   = np.arange(rows)
    col_grid   = np.arange(cols)

    def _base(ax, alpha=1.0):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax,
                  alpha=alpha, origin="upper", interpolation="nearest")

    def _mark_origins(ax, show_skin_corners=True):
        """
        Overlay the s=0 / s=1 frame-edge origins (stars) and, optionally,
        the skin-wall transition corners (hollow circles).

        Frame-edge origins (snapped onto the image border):
          ★ red          s = 0
          ★ deepskyblue  s = 1

        Skin transition corners (first/last skin contour point):
          ● red (faint)          skin_contour[0]
          ● deepskyblue (faint)  skin_contour[-1]
        """
        # Frame-edge stars
        ax.scatter(s_origin[1], s_origin[0],
                   marker="*", s=200, color="red",
                   edgecolors="white", linewidths=0.6,
                   zorder=8, label="_nolegend_")
        ax.scatter(s_end[1], s_end[0],
                   marker="*", s=200, color="deepskyblue",
                   edgecolors="white", linewidths=0.6,
                   zorder=8, label="_nolegend_")

        # Optionally draw a thin dashed line connecting the two frame origins,
        # illustrating the direction of the s-axis across the crop edge.
        if has_crop_edge:
            ax.plot([s_origin[1], s_end[1]],
                    [s_origin[0], s_end[0]],
                    color="white", linewidth=0.8, linestyle=":",
                    alpha=0.7, zorder=7)

        # Skin transition corners (smaller, semi-transparent)
        if show_skin_corners and len(skin_contour) >= 2:
            ax.scatter(skin_contour[0, 1], skin_contour[0, 0],
                       s=60, color="red", edgecolors="white",
                       linewidths=0.6, alpha=0.55, zorder=7)
            ax.scatter(skin_contour[-1, 1], skin_contour[-1, 0],
                       s=60, color="deepskyblue", edgecolors="white",
                       linewidths=0.6, alpha=0.55, zorder=7)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 1 — Boundary decomposition
    # ─────────────────────────────────────────────────────────────────────────
    _base(ax_img)
    ax_img.plot(boundary[:, 1], boundary[:, 0],
                color="lime", linewidth=1.5, zorder=3)
    if len(edge_contour):
        ax_img.scatter(edge_contour[:, 1], edge_contour[:, 0],
                       s=5, color="orange", zorder=4, linewidths=0)
    _mark_origins(ax_img, show_skin_corners=True)

    legend_handles = [leg_skin, leg_edge, leg_s0_frame, leg_s1_frame]
    if has_crop_edge:
        legend_handles += [leg_s0_skin, leg_s1_skin]
    ax_img.set_title("Boundary: Skin Wall vs Inferred Crop-Edge Segments",
                     fontsize=10)
    ax_img.legend(handles=legend_handles,
                  loc="upper right", fontsize=7.5, framealpha=0.75)
    ax_img.axis("off")

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 2 — Depth map
    # ─────────────────────────────────────────────────────────────────────────
    im_d = ax_d.imshow(d_display, cmap=cmap_d, vmin=0, vmax=1,
                       origin="upper", interpolation="nearest")
    cb_d = fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
    cb_d.set_label("d  (normalised depth to skin wall)", fontsize=9)
    cb_d.locator = MaxNLocator(nbins=5); cb_d.update_ticks()
    ax_d.imshow(edge_overlay, origin="upper", interpolation="nearest")
    ax_d.scatter(sr_c, sr_r, s=0.6, c="white", linewidths=0, alpha=0.7)
    ax_d.set_title(
        r"Depth Map  $d = d_\mathrm{raw}^{\,\mathrm{skin}}\,/\,d_\mathrm{max}$"
        "\nEDT seeded on skin ring  (frame pixels excluded as seeds)", fontsize=9)
    ax_d.legend(handles=[leg_skin, leg_edge], loc="upper right",
                fontsize=7, framealpha=0.75)
    ax_d.axis("off")

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 3 — Arc-length map
    # ─────────────────────────────────────────────────────────────────────────
    im_s = ax_s.imshow(s_display, cmap=cmap_s, vmin=0, vmax=1,
                       origin="upper", interpolation="nearest")
    cb_s = fig.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.04)
    cb_s.set_label("s  (normalised arc-length along skin wall)", fontsize=9)
    cb_s.locator = MaxNLocator(nbins=5); cb_s.update_ticks()
    ax_s.imshow(edge_overlay, origin="upper", interpolation="nearest")
    ax_s.scatter(sr_c, sr_r, s=0.6, c="white", linewidths=0, alpha=0.7)
    _mark_origins(ax_s, show_skin_corners=True)
    ax_s.set_title(
        r"Arc-length Map  $s = \ell(\mathbf{q}^*_{\!\mathrm{skin}})\,/\,S_\mathrm{total}$"
        "\nKD-tree on skin-wall points only  (crop pixels = orange)", fontsize=9)
    ax_s.legend(handles=[leg_skin, leg_edge, leg_s0_frame, leg_s1_frame],
                loc="upper right", fontsize=7, framealpha=0.75)
    ax_s.axis("off")

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 4 — Composite overlay
    # ─────────────────────────────────────────────────────────────────────────
    _base(ax_ov, alpha=0.72)
    cmap_d_obj = plt.get_cmap(cmap_d)
    cmap_s_obj = plt.get_cmap(cmap_s)

    ax_ov.contour(col_grid, row_grid, d_display,
                  levels=np.linspace(0.05, 0.95, n_iso_d),
                  colors=[cmap_d_obj(v) for v in np.linspace(0.05, 0.95, n_iso_d)],
                  linewidths=0.9, alpha=0.85)
    ax_ov.contour(col_grid, row_grid, s_display,
                  levels=np.linspace(0.02, 0.98, n_iso_s),
                  colors=[cmap_s_obj(v) for v in np.linspace(0.02, 0.98, n_iso_s)],
                  linewidths=0.9, linestyles="dashed", alpha=0.75)
    ax_ov.plot(boundary[:, 1], boundary[:, 0],
               color="lime", linewidth=1.2, alpha=0.9, zorder=3)
    if len(edge_contour):
        ax_ov.scatter(edge_contour[:, 1], edge_contour[:, 0],
                      s=5, color="orange", zorder=4, linewidths=0)
    _mark_origins(ax_ov, show_skin_corners=False)

    ov_legend = [
        Line2D([0], [0], color=cmap_d_obj(0.5), lw=1.5,
               label=f"iso-d  ({n_iso_d} levels, solid)"),
        Line2D([0], [0], color=cmap_s_obj(0.5), lw=1.5, linestyle="--",
               label=f"iso-s  ({n_iso_s} levels, dashed)"),
        leg_skin, leg_edge, leg_s0_frame, leg_s1_frame,
    ]
    ax_ov.legend(handles=ov_legend, loc="upper right",
                 fontsize=7.5, framealpha=0.75)
    ax_ov.set_title("Composite: iso-d (solid) and iso-s (dashed) overlay",
                    fontsize=11)
    ax_ov.axis("off")

    for cmap_name, label, pad in [(cmap_d, "d", 0.01), (cmap_s, "s", 0.07)]:
        sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=mcolors.Normalize(0, 1))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax_ov, fraction=0.025, pad=pad,
                          orientation="vertical")
        cb.set_label(label, fontsize=8)
        cb.locator = MaxNLocator(nbins=4); cb.update_ticks()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()
    return fig, axes
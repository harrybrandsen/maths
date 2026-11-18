"""
Neutron-density triangle: shale volume and porosity for a single log point.

End-members live in (NPHI, RHOB) space:
    matrix : clean matrix           (default quartz: NPHI = 0.00, RHOB = 2.65)
    fluid  : fluid-filled pore      (default water:  NPHI = 1.00, RHOB = 1.00)
    shale  : wet shale (user input)

Each log point is treated as a linear mix of the three end-members.
Solving for the volume fractions = computing barycentric coordinates:

    NPHI = Vm*phi_m + Vf*phi_f + Vsh*phi_sh
    RHOB = Vm*rho_m + Vf*rho_f + Vsh*rho_sh
    1    = Vm       + Vf       + Vsh
"""

import numpy as np


def nd_triangle(
    nphi,
    rhob,
    shale_point,
    matrix_point=(0.0, 2.65),
    fluid_point=(1.0, 1.0),
    phi_sh=None,
    plot=False,
):
    """
    Parameters
    ----------
    nphi, rhob : float
        Log point. NPHI in v/v (on the same lithology scale as matrix_point),
        RHOB in g/cc.
    shale_point : (nphi_sh, rhob_sh)
        Wet-shale end-member, e.g. (0.38, 2.45).
    matrix_point : (nphi_ma, rhob_ma)
        Clean matrix end-member. Default: quartz on a sandstone scale.
    fluid_point : (nphi_f, rhob_f)
        Pore-fluid end-member. Default: fresh water.
    phi_sh : float, optional
        Total porosity of the shale (bound water). Default: density
        porosity of the shale point, (rho_ma - rho_sh) / (rho_ma - rho_f).
    plot : bool
        If True, draw the triangle and the point (needs matplotlib).

    Returns
    -------
    dict with
        v_matrix, v_shale, v_fluid : volume fractions (sum to 1)
        phi_effective : porosity of the non-shale part (= v_fluid)
        phi_total     : v_fluid + v_shale * phi_sh
        inside        : True if the point lies inside the triangle
        raw_fractions : unclipped barycentric coordinates (matrix, fluid, shale)
    """
    nphi_ma, rho_ma = matrix_point
    nphi_f, rho_f = fluid_point
    nphi_sh, rho_sh = shale_point

    # Columns = end-members (matrix, fluid, shale); rows = NPHI, RHOB, sum-to-one
    A = np.array([
        [nphi_ma, nphi_f, nphi_sh],
        [rho_ma,  rho_f,  rho_sh],
        [1.0,     1.0,    1.0],
    ])
    if abs(np.linalg.det(A)) < 1e-12:
        raise ValueError("End-members are collinear; the triangle is degenerate.")
    b = np.array([nphi, rhob, 1.0])
    v_ma, v_f, v_sh = np.linalg.solve(A, b)

    tol = 1e-9
    inside = all(v >= -tol for v in (v_ma, v_f, v_sh))

    raw = (v_ma, v_f, v_sh)

    # Clip to [0, 1] and renormalise so the fractions stay physical
    clipped = np.clip(raw, 0.0, 1.0)
    clipped = clipped / clipped.sum()
    v_ma_c, v_f_c, v_sh_c = clipped

    if phi_sh is None:
        phi_sh = (rho_ma - rho_sh) / (rho_ma - rho_f)

    result = {
        "v_matrix": float(v_ma_c),
        "v_shale": float(v_sh_c),
        "v_fluid": float(v_f_c),
        "phi_effective": float(v_f_c),
        "phi_total": float(v_f_c + v_sh_c * phi_sh),
        "inside": bool(inside),
        "raw_fractions": tuple(float(v) for v in raw),
    }

    if plot:
        _plot(nphi, rhob, matrix_point, fluid_point, shale_point, result)

    return result


def _plot(nphi, rhob, matrix_point, fluid_point, shale_point, res):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    pts = np.array([matrix_point, fluid_point, shale_point, matrix_point])
    ax.plot(pts[:, 0], pts[:, 1], "k-", lw=1.5)

    for (x, y), name in zip([matrix_point, fluid_point, shale_point],
                            ["Matrix", "Fluid", "Shale"]):
        ax.plot(x, y, "ko")
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6))

    ax.plot(nphi, rhob, "r*", ms=14, label="Log point")
    ax.set_xlabel("NPHI (v/v)")
    ax.set_ylabel("RHOB (g/cc)")
    ax.invert_yaxis()  # density increases downward, porosity up/right
    ax.grid(alpha=0.3)
    ax.set_title(
        f"Vsh = {res['v_shale']:.2f}, "
        f"phi_e = {res['phi_effective']:.2f}, "
        f"phi_t = {res['phi_total']:.2f}"
        + ("" if res["inside"] else "  (outside triangle)")
    )
    ax.legend()
    plt.show()


if __name__ == "__main__":
    out = nd_triangle(
        nphi=0.25,
        rhob=2.30,
        shale_point=(0.38, 2.45),
        plot=False,
    )
    for k, v in out.items():
        print(f"{k:14s} {v}")

import numpy as np
from scipy.interpolate import CubicSpline
import torch

def pd_fv_syms():
    # finite volume implementation

    model = {}
    model['param'] = 'log'
    model['forward'] = True
    model['adjoint'] = False

    # PDE discretization
    n_grid = 300
    grid = np.linspace(0, 1, n_grid)

    h2inv = (1 / (grid[1] - grid[0])) ** 2

    grid_x = np.linspace(grid[0] + (grid[1] - grid[0]) / 2, grid[-2] + (grid[-1] - grid[-2]) / 2, n_grid - 1)

    # STATES
    # create state syms: vector for x at each grid point
    x = torch.zeros(n_grid - 1, dtype=torch.float64, requires_grad=True)

    # PARAMETERS ( for these sensitivities will be computed )
    # create parameter syms
    D = torch.zeros(9, dtype=torch.float64, requires_grad=True)
    v = torch.zeros(9, dtype=torch.float64, requires_grad=True)
    a = torch.zeros(9, dtype=torch.float64, requires_grad=True)

    p = [D, v, a]

    # Define Db1
    spline_x = np.arange(0, 1.125, 0.125)
    spline_y = np.log([D1, D2, D3, D4, D5, D6, D7, D8, D9])
    spline = CubicSpline(spline_x, spline_y)
    Db1 = torch.exp(torch.tensor(spline(grid[1:-1]), dtype=torch.float64))

    # Define vb1
    spline_y_v = np.log([v1, v2, v3, v4, v5, v6, v7, v8, v9])
    spline_v = CubicSpline(spline_x, spline_y_v)
    vb1 = torch.exp(torch.tensor(spline_v(grid[1:-1]), dtype=torch.float64))
    dvb1dgrid = vb1 * torch.tensor(spline_v(grid[1:-1], 1), dtype=torch.float64)

    grid0 = 29
    v_end_b1 = CubicSpline(grid[-grid0:], vb1[-grid0:], bc_type='clamped')(grid[-1])
    vb1[-grid0:] = v_end_b1

    # Define ab1
    spline_y_a = np.log([a1, a2, a3, a4, a5, a6, a7, a8, a9])
    spline_a = CubicSpline(spline_x, spline_y_a)
    ab1 = torch.tensor(spline_a(grid_x), dtype=torch.float64)

    # SYSTEM EQUATIONS
    # create symbolic variable for time
    t = torch.tensor(0, dtype=torch.float64)

    xdot = torch.zeros_like(x)

    # dynamic branch 1
    xdot[0] = h2inv * (-Db1[0] * (x[0] - x[1])) - vb1[0] * 1 / 2 * (x[0] + x[1]) * torch.sqrt(h2inv) + ab1[0] * x[0]

    xdot[-1] = h2inv * (Db1[-1] * (x[-2] - x[-1])) + vb1[-1] * 1 / 2 * (
                x[-2] + x[-1]) * torch.sqrt(h2inv) + ab1[-1] * x[-1]

    for i in range(1, n_grid - 2):
        xdot[i] = h2inv * (Db1[i - 1] * (x[i - 1] - x[i]) - Db1[i] * (x[i] - x[i + 1])) + torch.sqrt(
            h2inv) * 1 / 2 * (
                          vb1[i - 1] * (x[i - 1] + x[i]) - vb1[i] * (x[i] + x[i + 1])) + ab1[i] * x[i]

    # INITIAL CONDITIONS
    x0 = torch.zeros_like(x)
    k = torch.tensor(np.linspace(0, 1, n_grid - 1), dtype=torch.float64)
    x0 = k

    # OBSERVABLES
    N = torch.sum(x / torch.sqrt(h2inv))
    y = torch.cat((x, torch.unsqueeze(N, dim=0)))

    # SYSTEM STRUCT
    model['sym'] = {}
    model['sym']['x'] = x
    model['sym']['k'] = k
    model['sym']['xdot'] = xdot
    model['sym']['p'] = p
    model['sym']['x0'] = x0
    model['sym']['y'] = y

    return model

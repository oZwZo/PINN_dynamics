def simulate_pd_branching_fv_toy(tout, theta, kappa=[], data=None, options=None):
    if options is None:
        options = {
            'atol': None,
            'rtol': None,
            'maxsteps': None,
            'tstart': None,
            'sens_ind': range(1, 39),
            'x0': None,
            'sx0': None,
            'lmm': 2,
            'iter': 2,
            'linsol': 1,
            'stldet': 1,
            'qPositiveX': [],
            'sensi_meth': 'forward',
            'adjoint': False,
            'ism': 1,
            'Nd': 1000,
            'interpType': 1,
            'ordering': 0
        }

    chainRuleFactor = theta[options['sens_ind']]

    if 'pbar' in options and options['pbar'] is not None:
        pbar = options['pbar']
    else:
        pbar = np.ones_like(theta)

    if 'ss' in options and options['ss'] > 0:
        if options['sensi'] > 1:
            raise ValueError('Computation of steady state sensitivity only possible for first order sensitivities')
        options['sensi'] = 0

    if options['sensi'] > 0:
        if options['sensi_meth'] == 2:
            raise ValueError('adjoint sensitivities are disabled as necessary routines were not compiled')

    np = len(options['sens_ind'])
    if np == 0:
        options['sensi'] = 0

    nxfull = 529

    if 'qPositiveX' not in options:
        options['qPositiveX'] = np.zeros(nxfull)
    else:
        options['qPositiveX'] = np.asarray(options['qPositiveX'])
        if len(options['qPositiveX']) < nxfull:
            raise ValueError('provided condition vector is too short')

    plist = np.array(options['sens_ind']) - 1

    if data is not None and isinstance(data, amidata):
        if data.ne > 0:
            options['nmaxevent'] = data.ne
        else:
            data.ne = options['nmaxevent']

        if kappa == []:
            kappa = data.condition
        if tout == []:
            tout = data.t
    else:
        data = []

    if not np.all(tout == np.sort(tout)):
        raise ValueError('Provided time vector is not monotonically increasing!')

    if len(tout) != len(set(tout)):
        raise ValueError('Provided time vector has non-unique entries!!')

    if max(options['sens_ind']) > 38:
        raise ValueError('Sensitivity index exceeds parameter dimension!')

    init = {}

    if 'x0' in options and options['x0'] is not None:
        if options['x0'].shape[1] != 1:
            raise ValueError('x0 field must be a column vector!')
        if options['x0'].shape[0] != nxfull:
            raise ValueError('Number of rows in x0 field does not agree with number of states!')
        init['x0'] = options['x0']

    if 'sx0' in options and options['sx0'] is not None:
        if options['sx0'].shape[1] != np:
            raise ValueError('Number of rows in sx0 field does not agree with number of model parameters!')
        if options['sx0'].shape[0] != nxfull:
            raise ValueError('Number of columns in sx0 field does not agree with number of states!')
        init['sx0'] = options['sx0'] * (1 / chainRuleFactor)

    sol = ami_pd_branching_fv_toy(tout, theta[:38], kappa[:529], options, plist, pbar, [], init, data)

    if options['sensi'] == 1:
        sol.sllh = sol.sllh * theta[options['sens_ind']]
        sol.sx = sol.sx * theta[options['sens_ind']].reshape(1, -1, 1)
        sol.sy = sol.sy * theta[options['sens_ind']].reshape(1, -1, 1)

    if options['sensi_meth'] == 3:
        sol.dxdotdp = sol.dxdotdp * theta[options['sens_ind']].reshape(1, -1)
        sol.dydp = sol.dydp * theta[options['sens_ind']].reshape(1, -1)
        sol.sx = -sol.J @ sol.dxdotdp
        sol.sy = sol.dydx @ sol.sx + sol.dydp

    return sol
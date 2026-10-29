"""Stand-in optimizer source for the dashboard code panel.

This file only shows text. It does not run. It uses the same step markers
as topopt/optimize.py (CONTRACTS.md section 6).
"""


def optimize(problem, settings, writer):
    # [step:setup]
    rho = initial_density(problem)
    mma = MMA(n=rho.size, m=1)
    solver = LinearSolver("cholesky")
    for it in range(1, settings.iterations + 1):
        # [step:assemble]
        E_e = E_min + rho_phys**p * (E0 - E_min)
        K = assembler.K(E_e)
        # [step:solve]
        solver.factor(K)
        U = solver.solve(F)
        # [step:sensitivity]
        ce = np.einsum("ij,jk,ik->i", U[edof], ke, U[edof])
        dc = -p * rho_phys ** (p - 1) * ce
        # [step:filter]
        dc = filter_T(dc * dproj)
        dv = filter_T(dvol * dproj)
        # [step:update]
        rho = mma.step(rho, c, dc, vol - vf, dv)
        rho_phys = project(filter_(rho), beta)
        # [step:snapshot]
        writer.snapshot(it, rho_phys)
        writer.history(it, c, vol)
    # [end]
    return rho_phys

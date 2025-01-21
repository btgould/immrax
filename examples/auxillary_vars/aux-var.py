import pickle
import time
from typing import Literal
import time 

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import immrax as irx
from immrax.embedding import AuxVarEmbedding
from immrax.inclusion import interval, mjacif
from immrax.utils import (
    angular_sweep,
    run_times,
    draw_trajectory_2d,
    draw_refined_trajectory_2d,
)

# Read papers about contraction and stability


class HarmOsc(irx.System):
    def __init__(self) -> None:
        self.evolution = "continuous"
        self.xlen = 2
        self.name = "Harmonic Oscillator"

    def f(self, t, x: jnp.ndarray) -> jnp.ndarray:
        x1, x2 = x.ravel()
        return jnp.array([-x2, x1])


class VanDerPolOsc(irx.System):
    def __init__(self, mu: float = 1) -> None:
        self.evolution = "continuous"
        self.xlen = 2
        self.name = "Van der Pol Oscillator"
        self.mu = mu

    def f(self, t, x: jnp.ndarray) -> jnp.ndarray:
        x1, x2 = x.ravel()
        return jnp.array([self.mu * (x1 - 1 / 3 * x1**3 - x2), x1 / self.mu])


def show_refinements(mode: Literal["sample", "linprog"]):
    fig = plt.figure()
    # fig, axs = plt.subplots(int(jnp.ceil(N / 3)), 3, figsize=(5, 5))
    fig.suptitle(f"Reachable Sets of the {sys.name}")
    plt.gca().set_xlabel(r"$x_1$")
    plt.gca().set_ylabel(r"$x_2$")
    # axs = axs.reshape(-1)

    H = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    for i in range(len(aux_vars)):
        # Add new refinement
        print(f"Adding auxillary variable {aux_vars[i]}")
        H = jnp.append(H, jnp.array([aux_vars[i]]), axis=0)
        lifted_x0_int = interval(H) @ x0_int

        if i < N - 1:
            continue

        # Compute refined trajectory
        auxsys = AuxVarEmbedding(sys, H, mode=mode, if_transform=mjacif)
        # auxsys.compute_trajectory(0.0, sim_len, irx.i2ut(lifted_x0_int)) # can't use compute_trajectory because scipy LP is not traceable

        # Euler integrate instead
        start = time.time()
        dt = 0.01
        ts = jnp.arange(0.0, sim_len, dt)
        x = irx.i2ut(lifted_x0_int)
        ys = jnp.empty((ts.size, x.size))
        for i, t in enumerate(ts):
            ys = ys.at[i].set(x)
            x += dt * auxsys.E(t, x)

        traj = Trajectory(ts, ys, jnp.ones_like(ts, dtype=bool))
        print(
            f"Computing trajectory with {mode} refinement for {i + 1} aux vars took {time.time() - start:.4g}s"
        )

        ys_int = [irx.ut2i(y) for y in traj.ys]
        print(f"Final bound: \n{ys_int[-1][:2]}")

        # Display results
        # plt.sca(axs[i])
        # axs[i].set_title(rf"$\theta = {i+1} \frac{{\pi}}{{{N+1}}}$")
        # plt.gca().set_title(rf"$\theta = {i + 1} \frac{{\pi}}{{{N + 1}}}$")
        draw_refined_trajectory_2d(traj, H)


x0_int = irx.icentpert(jnp.array([1.0, 0.0]), jnp.array([0.1, 0.1]))
sim_len = 0.628

# Certain values of N are not good choices, as they will generate angle theta=pi/2, which is redundant with the actual state vars
N = 6
aux_vars = angular_sweep(N)

plt.rcParams.update({"text.usetex": True, "font.family": "CMU Serif", "font.size": 14})
plt.figure()


# Trajectory of unrefined system
sys = VanDerPolOsc()  # Can use an arbitrary system here
embsys = irx.mjacemb(sys)
traj = embsys.compute_trajectory(
    0.0,
    sim_len,
    irx.i2ut(x0_int),
)
plt.gcf().suptitle(f"{sys.name} with Uncertainty (No Refinement)")
draw_trajectory_2d(traj)


# show_refinements("sample")
show_refinements("linprog")

print("Plotting finished")
plt.show()

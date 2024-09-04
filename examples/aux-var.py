import immrax as irx
from immrax.embedding import AuxVarEmbedding
from immrax.utils import draw_iarray
import jax.numpy as jnp
import matplotlib.pyplot as plt

x0 = irx.icentpert(jnp.array([1, 0, 1]), jnp.array([0.1, 0.1, 0.2]))
H = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


class HarmOsc(irx.System):
    def __init__(self) -> None:
        self.evolution = "continuous"
        self.xlen = 2

    def f(self, t, x: jnp.ndarray) -> jnp.ndarray:
        x1, x2 = x.ravel()
        return jnp.array([-x2, x1])


# Create UT embedded system
embsys = AuxVarEmbedding(HarmOsc(), H)

# Compute trajectory in embedding
traj = embsys.compute_trajectory(
    0.0,
    1.56,
    irx.i2ut(x0),
)

# Clean up and display results
tfinite = jnp.where(jnp.isfinite(traj.ts))
ts_clean = traj.ts[tfinite]
ys_clean = traj.ys[tfinite]

ax = plt.gca()
y_int = [irx.ut2i(y) for y in ys_clean]
for timestep, bound in zip(ts_clean, y_int):
    draw_iarray(ax, bound)

plt.show()

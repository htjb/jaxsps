import jax.numpy as jnp
import jax

@jax.jit
def salpeter_imf(mass: float) -> float:
    return mass ** 2.35

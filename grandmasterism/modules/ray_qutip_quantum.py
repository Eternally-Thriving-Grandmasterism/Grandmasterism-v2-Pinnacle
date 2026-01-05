"""
Ray + QuTiP Distributed Quantum Simulations — Parallel Thriving Eternal
Ray tasks for mesolve/sesolve parameter sweeps, quantum dynamics batches
"""

import ray
import qutip as qt
import numpy as np

ray.init(ignore_reinit_error=True)

@ray.remote
def remote_mesolve_sweep(gamma: float, N: int = 10, tmax: float = 10.0):
    """Remote task: Cavity decay master equation solve"""
    tlist = np.linspace(0, tmax, 100)
    a = qt.destroy(N)
    H = a.dag() * a
    c_ops = [np.sqrt(gamma) * a]
    psi0 = qt.basis(N, 5)
    result = qt.mesolve(H, psi0, tlist, c_ops, [a.dag() * a])
    return {
        "gamma": gamma,
        "final_expect": result.expect[0][-1],
        "tlist": tlist.tolist(),
        "expect_values": result.expect[0].tolist()
    }

def ray_parallel_quantum_sweep(gammas: list = [0.1, 0.5, 1.0]) -> list:
    """Parallel mesolve sweeps with Ray tasks"""
    futures = [remote_mesolve_sweep.remote(g) for g in gammas]
    results = ray.get(futures)
    print(f"Ray-QuTiP parallel quantum sweep complete — {len(results)} simulations thriving distributed eternal!")
    return results

@ray.remote
class QuantumStateActor:
    """Ray actor: Persistent quantum state for sequential evolutions"""
    def __init__(self, N: int = 15):
        self.N = N
        self.a = qt.destroy(N)
        self.state = qt.basis(N, 8)
        print(f"Ray Quantum State Actor initialized — persistent thriving state eternal for N={N}.")

    def evolve_decay(self, gamma: float, tmax: float = 5.0):
        tlist = np.linspace(0, tmax, 50)
        H = self.a.dag() * self.a
        c_ops = [np.sqrt(gamma) * self.a]
        result = qt.mesolve(H, self.state, tlist, c_ops, [self.a.dag() * self.a])
        self.state = result.states[-1]
        return {"final_expect": result.expect[0][-1], "thriving_state": "persistent_decay_evolved"}

def ray_quantum_state_demo(gammas: list = [0.2, 0.8]) -> list:
    """Distributed persistent state evolution with Ray actor"""
    actor = QuantumStateActor.remote(N=20)
    futures = [actor.evolve_decay.remote(g) for g in gammas]
    results = ray.get(futures)
    print(f"Ray Quantum State Actor demo complete — persistent thriving evolution eternal!")
    return results

if __name__ == "__main__":
    sweeps = ray_parallel_quantum_sweep([0.3, 0.7, 1.2])
    print(sweeps)
    state_demo = ray_quantum_state_demo([0.4, 1.0])
    print(state_demo)

import streamlit as st
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="P452 Project 2: Many-Body Simulation", layout="wide")

st.title("P452 Project 2: Simulation of Many-Body System")
st.write("**Lecturer:** Cheng Chin | **Due:** May 7, 2026")

# --- Sidebar Navigation ---
page = st.sidebar.radio("Navigation", ["Phase 1: Heisenberg Model", "Phase 2: Bose-Fermi Mixtures"])

# ==========================================
# PHASE 1: HEISENBERG MODEL
# ==========================================
if page == "Phase 1: Heisenberg Model":
    st.header("Phase 1: Exact Diagonalization of the Heisenberg Model")

    # --- Markdown Analysis for Phase 1 ---
    st.markdown(r"""
    ### Checkpoint 1.1: The 2-Site Dimer and Spin Frustration (Pen and Paper)

    **1. Hilbert Space**
    For $N=2$ spins with $S=1/2$, each site has 2 states (up or down). The total dimension of the Hilbert space is $2^2 = 4$. 
    The basis states are: $|\uparrow\uparrow\rangle$, $|\uparrow\downarrow\rangle$, $|\downarrow\uparrow\rangle$, $|\downarrow\downarrow\rangle$.

    **2. Hamiltonian Matrix**
    The interaction is $\hat{S}_1 \cdot \hat{S}_2 = \hat{S}^z_1 \hat{S}^z_2 + \frac{1}{2}(\hat{S}^+_1 \hat{S}^-_2 + \hat{S}^-_1 \hat{S}^+_2)$. 
    Let's apply the Hamiltonian to our basis states:
    * $\hat{H}|\uparrow\uparrow\rangle = \left(\frac{J}{4} + H\right)|\uparrow\uparrow\rangle$
    * $\hat{H}|\downarrow\downarrow\rangle = \left(\frac{J}{4} - H\right)|\downarrow\downarrow\rangle$
    * $\hat{H}|\uparrow\downarrow\rangle = -\frac{J}{4}|\uparrow\downarrow\rangle + \frac{J}{2}|\downarrow\uparrow\rangle$
    * $\hat{H}|\downarrow\uparrow\rangle = -\frac{J}{4}|\downarrow\uparrow\rangle + \frac{J}{2}|\uparrow\downarrow\rangle$

    This gives the matrix in the ordered basis ($|\uparrow\uparrow\rangle$, $|\uparrow\downarrow\rangle$, $|\downarrow\uparrow\rangle$, $|\downarrow\downarrow\rangle$):
    $$H = \begin{pmatrix} \frac{J}{4} + H & 0 & 0 & 0 \\ 0 & -\frac{J}{4} & \frac{J}{2} & 0 \\ 0 & \frac{J}{2} & -\frac{J}{4} & 0 \\ 0 & 0 & 0 & \frac{J}{4} - H \end{pmatrix}$$

    **3. Ground State Transition**
    Diagonalizing the $2 \times 2$ inner block yields the Singlet $|S\rangle$ and Triplet $|T_0\rangle$ states. The eigenenergies are:
    * $E_{T+} = \frac{J}{4} + H$  (from $|\uparrow\uparrow\rangle$)
    * $E_{T-} = \frac{J}{4} - H$  (from $|\downarrow\downarrow\rangle$)
    * $E_S = -\frac{3J}{4}$ (Singlet: $\frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle - |\downarrow\uparrow\rangle)$)
    * $E_{T0} = \frac{J}{4}$ (Triplet 0: $\frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle + |\downarrow\uparrow\rangle)$)

    At $H=0$, the ground state is the Singlet ($E = -3J/4$), representing an anti-ferromagnetic configuration. Assuming $H < 0$ (such that the field points "up" and lowers the energy of $|\uparrow\uparrow\rangle$), the critical field $H_c$ occurs when the fully polarized state crosses the singlet:
    $$\frac{J}{4} - |H_c| = -\frac{3J}{4} \implies |H_c| = J$$

    **4. Spin Frustration (3 Spins on a Ring)**
    * **Symmetry Reduction:** The total Hilbert space dimension is $2^3 = 8$. Because $[\hat{H}, \hat{S}^z_{\text{total}}] = 0$, the Hamiltonian cannot flip the total number of up/down spins. The matrix block-diagonalizes into sectors: $S^z = 3/2$ (1 state), $1/2$ (3 states), $-1/2$ (3 states), and $-3/2$ (1 state).
    * **Calculation:** * **Sector +3/2** ($|\uparrow\uparrow\uparrow\rangle$): $E = \frac{3J}{4} + \frac{3H}{2}$.
        * **Sector +1/2** (Basis: $|\uparrow\uparrow\downarrow\rangle$, $|\uparrow\downarrow\uparrow\rangle$, $|\downarrow\uparrow\uparrow\rangle$): Each state has one favorable AF bond and two unfavorable FM bonds. The diagonal elements are $-\frac{J}{4} + \frac{H}{2}$. The off-diagonal ladder operators flip anti-aligned neighbors with amplitude $J/2$. The block is:
        $$H_{1/2} = \begin{pmatrix} -\frac{J}{4} + \frac{H}{2} & \frac{J}{2} & \frac{J}{2} \\ \frac{J}{2} & -\frac{J}{4} + \frac{H}{2} & \frac{J}{2} \\ \frac{J}{2} & \frac{J}{2} & -\frac{J}{4} + \frac{H}{2} \end{pmatrix}$$
        The eigenvalues of this block are $E = \frac{3J}{4} + \frac{H}{2}$ (non-degenerate) and $E = -\frac{3J}{4} + \frac{H}{2}$ (doubly degenerate).
    * **Analysis:** At $H=0$, the ground state is in the $\pm 1/2$ sectors with $E = -3J/4$. It is 4-fold degenerate. At high $H \gg J$, the field dominates, and the fully polarized state $|\downarrow\downarrow\downarrow\rangle$ becomes the unique ground state. The 4-fold degeneracy at $H=0$ physically represents **frustration**: on a triangle, you cannot anti-align all three spins simultaneously without paying an energy penalty.

    ---
    ### Checkpoint 1.2: From 2x2 to 6x6 Square Lattices

    **1. Hilbert Space and Dimension**
    For a $2 \times 2$ lattice, $N=4$. Total dimension = $2^4 = 16$. 
    For the $S^z_{\text{total}} = 0$ sector (2 up, 2 down), the dimension is $D = \binom{4}{2} = 6$.
    The basis states are: $|\uparrow\uparrow\downarrow\downarrow\rangle$, $|\uparrow\downarrow\uparrow\downarrow\rangle$, $|\uparrow\downarrow\downarrow\uparrow\rangle$, $|\downarrow\uparrow\uparrow\downarrow\rangle$, $|\downarrow\uparrow\downarrow\uparrow\rangle$, $|\downarrow\downarrow\uparrow\uparrow\rangle$.

    **2. Role of the External Field**
    The Zeeman term is $\hat{H}_{\text{Zeeman}} = H \sum \hat{S}^z_i$. Since the operator $\sum \hat{S}^z_i = \hat{S}^z_{\text{total}}$, for any state $|\phi\rangle$ in the $S^z_{\text{total}} = 0$ sector, $\hat{S}^z_{\text{total}}|\phi\rangle = 0$. 
    Therefore, $H \sum \hat{S}^z_i |\phi\rangle = 0$. The external field only shifts the overall energy of sectors with net magnetization; it does not split or shift levels *within* the zero-magnetization block.

    **3. Eigenenergies and Limiting Regimes**
    * **$H=0$ or $J=0$:** At $J=0$, spins don't interact. At $H=0$, $J > 0$, the system minimizes energy via anti-ferromagnetic coupling. The ground state lies in the $S^z_{\text{total}} = 0$ sector. A classical Neel state (up-down-up-down) has energy $-J$. Quantum fluctuations further lower the quantum ground state energy below the classical value.
    * **General case $H, J > 0$:** As $H$ increases, it tilts the energy of sectors with higher magnetization downwards. The ground state transitions from $S^z = 0 \to 1 \to 2$. These critical fields $H_c$ create the "magnetization staircase." In the thermodynamic limit ($N \to \infty$), the magnetization steps smooth out into a continuous phase transition to the Ferromagnetic (FM) phase.
    """)

    # --- Interactive Code UI for Phase 1 ---
    st.markdown("---")
    st.subheader("Interactive Numerical Simulation: Magnetization Staircase")
    
    col1, col2 = st.columns(2)
    with col1:
        N_selected = st.selectbox("Number of Spins (N)", [4, 16], index=0, help="N=16 may take a few seconds to diagonalize.")
        J_val = st.slider("Anti-ferromagnetic Coupling (J)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    with col2:
        H_max = st.slider("Max Magnetic Field (H)", min_value=1.0, max_value=20.0, value=10.0, step=1.0)
        num_points = st.slider("Resolution (Points)", min_value=10, max_value=100, value=40)

    # Core Logic
    @st.cache_data
    def get_basis_sz(N, k):
        basis = []
        for i in range(2**N):
            if bin(i).count('1') == k:
                basis.append(i)
        return basis

    @st.cache_data
    def build_hamiltonian(N, J, H_field, Sz_target=None):
        L = int(np.sqrt(N))
        bonds = []
        for i in range(L):
            for j in range(L):
                idx = i * L + j
                right = i * L + (j + 1) % L
                down = ((i + 1) % L) * L + j
                bonds.extend([(idx, right), (idx, down)])

        if Sz_target is None:
            basis = list(range(2**N))
        else:
            k = int(N/2 + Sz_target)
            basis = get_basis_sz(N, k)
            
        dim = len(basis)
        state_to_idx = {state: i for i, state in enumerate(basis)}
        
        rows, cols, data = [], [], []
        
        for i, state in enumerate(basis):
            sz_total = sum(0.5 if (state & (1 << j)) else -0.5 for j in range(N))
            rows.append(i); cols.append(i); data.append(H_field * sz_total)
            
            for s1, s2 in bonds:
                sz1 = 0.5 if (state & (1 << s1)) else -0.5
                sz2 = 0.5 if (state & (1 << s2)) else -0.5
                rows.append(i); cols.append(i); data.append(J * sz1 * sz2)
                
                if sz1 != sz2:
                    new_state = state ^ ((1 << s1) | (1 << s2))
                    if new_state in state_to_idx:
                        j = state_to_idx[new_state]
                        rows.append(i); cols.append(j); data.append(J / 2.0)
                        
        return sp.csr_matrix((data, (rows, cols)), shape=(dim, dim))

    if st.button("Run ED Simulation"):
        with st.spinner(f"Calculating Exact Diagonalization for N={N_selected}..."):
            H_vals = np.linspace(0, H_max, num_points)
            magnetizations = []
            
            for H_field in H_vals:
                min_E = float('inf')
                gs_mag = 0
                for Sz in range(-N_selected//2, N_selected//2 + 1):
                    H_mat = build_hamiltonian(N_selected, J_val, H_field, Sz_target=Sz)
                    
                    if H_mat.shape[0] <= 2:
                        evals = H_mat.toarray().diagonal()
                    else:
                        evals = sla.eigsh(H_mat, k=1, which='SA', return_eigenvectors=False)
                        
                    E0 = np.min(evals)
                    if E0 < min_E:
                        min_E = E0
                        gs_mag = Sz / (N_selected/2)
                        
                magnetizations.append(abs(gs_mag)) 

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(H_vals / J_val, magnetizations, label=f"{int(np.sqrt(N_selected))}x{int(np.sqrt(N_selected))} (N={N_selected})", marker='o', linestyle='-')
            ax.set_xlabel('H / J')
            ax.set_ylabel(r'Normalized Magnetization $\langle M_z \rangle$')
            ax.set_title(f'Magnetization Plateaus (N={N_selected})')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)


# ==========================================
# PHASE 2: BOSE-FERMI MIXTURES
# ==========================================
elif page == "Phase 2: Bose-Fermi Mixtures":
    st.header("Phase 2: Hydrodynamics of Bose-Fermi Mixtures")

    # --- Markdown Analysis for Phase 2 ---
    st.markdown(r"""
    ### Checkpoint 2.1: Uniform Gas in the Ground State

    **1. Bosonic Chemical Potential**
    In the Thomas-Fermi approximation, we neglect the kinetic term $-\frac{\hbar^2 \nabla^2}{2m_B}$. 
    Without fermions ($a_{BF}=0$) and trap ($V_B=0$), the GP equation reduces to:
    $$\mu_B = g_B |\Psi_B|^2 = g_B n_B$$
    Compressibility is $\kappa \propto \left(\frac{\partial \mu_B}{\partial n_B}\right)^{-1}$. Since $\frac{\partial \mu_B}{\partial n_B} = g_B = \frac{4\pi \hbar^2 a_B}{m_B}$, as long as $a_B > 0$ (repulsive bosons), compressibility is positive, and the BEC is stable against collapse.

    **2. Fermionic "Interaction"**
    For a uniform Fermi gas, $\mu_F = E_F = \frac{\hbar^2}{2m_F}(6\pi^2 n_F)^{2/3}$. 
    In the hydrodynamic equation, the term $K_F(n_F)$ must equate to this chemical potential to ensure the system rests at the Fermi energy. Thus:
    $$K_F(n_F) = \frac{\hbar^2}{2m_F}(6\pi^2 n_F)^{2/3}$$
    **Scaling difference:** The bosonic mean-field term scales linearly with density ($n_B^1$), while the fermionic Pauli pressure term scales as $n_F^{2/3}$.

    **3. Interspecies Coupling**
    With $a_{BF} \neq 0$, the uniform coupled equations become:
    $$\mu_B = g_B n_B + g_{BF} n_F$$
    $$\mu_F = E_F(n_F) + g_{BF} n_B$$

    By taking $n_B = (\mu_B - g_{BF}n_F) / g_B$ and substituting it into the fermionic equation:
    $$\mu_F = E_F(n_F) + \frac{g_{BF}\mu_B}{g_B} - \frac{g_{BF}^2}{g_B} n_F$$
    This creates an effective fermionic interaction $g_{F,\text{eff}} = - \frac{g_{BF}^2}{g_B}$. Notice it is *always negative* (attractive) regardless of the sign of $g_{BF}$.

    **Stability Analysis:**
    The mixture is stable if $\text{det}(M) > 0$:
    $$M = \begin{pmatrix} g_B & g_{BF} \\ g_{BF} & \frac{\partial E_F}{\partial n_F} \end{pmatrix}$$
    $$\text{det}(M) = g_B \left( \frac{\hbar^2}{3m_F}(6\pi^2)^{2/3} n_F^{-1/3} \right) - g_{BF}^2 > 0$$

    ---
    ### Checkpoint 2.2: Mixtures in a Trap

    **1 & 2. Thomas-Fermi and LDA Limits**
    From the coupled equations including the harmonic trap $V_i(r) = \frac{1}{2}m_i \omega_i^2 r^2$:
    * **Bosons:** $n_B(r) = \max \left( 0, \frac{\mu_B - V_B(r) - g_{BF}n_F(r)}{g_B} \right)$
    * **Fermions:** $n_F(r) = \max \left( 0, \frac{1}{6\pi^2} \left[ \frac{2m_F}{\hbar^2} (\mu_F - V_F(r) - g_{BF}n_B(r)) \right]^{3/2} \right)$

    **3. Physical Intuition**
    * **Attractive ($g_{BF} < 0$):** The $-g_{BF} n_F(r)$ term increases the effective local chemical potential for bosons where fermions are dense, and vice versa. They mutually attract, pulling each other into a localized, dense core at the trap center. 
    * **Repulsive ($g_{BF} > 0$):** The species repel. If $g_{BF}$ is large enough (violating stability), they undergo **phase separation**. The denser gas (usually the BEC) sits at the center, acting as a "plunger" pushing the fermions into a shell at the edges of the trap.
    """)

    # --- Interactive Code UI for Phase 2 ---
    st.markdown("---")
    st.subheader("Interactive Numerical Simulation: Coupled Hydrodynamic Density Profiles")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        g_BF = st.slider("Interspecies Coupling (g_BF)", min_value=-5.0, max_value=25.0, value=0.0, step=0.5, 
                         help="Negative = Attractive, Positive = Repulsive. High Positive = Phase Separation.")
        g_B = st.number_input("Boson Repulsion (g_B)", value=10.0)
    with col2:
        m_B = st.number_input("Boson Mass (m_B)", value=1.0)
        m_F = st.number_input("Fermion Mass (m_F)", value=0.46) # e.g. K/Rb ratio
    with col3:
        mu_B = st.number_input("Boson Chem. Potential (mu_B)", value=15.0)
        mu_F = st.number_input("Fermion Chem. Potential (mu_F)", value=10.0)

    # Core Logic
    def solve_densities(g_BF_val, max_iter=500, mix=0.1):
        r = np.linspace(0, 10, 500)
        w_B, w_F = 1.0, 1.0
        V_B = 0.5 * m_B * w_B**2 * r**2
        V_F = 0.5 * m_F * w_F**2 * r**2
        
        n_B = np.maximum(0, (mu_B - V_B) / g_B)
        n_F = np.zeros_like(r)
        
        for _ in range(max_iter):
            # LDA for Fermions
            eff_mu_F = mu_F - V_F - g_BF_val * n_B
            new_n_F = np.maximum(0, (1.0 / (6 * np.pi**2)) * (2 * m_F * np.maximum(0, eff_mu_F))**1.5)
            
            # TF for Bosons
            eff_mu_B = mu_B - V_B - g_BF_val * new_n_F
            new_n_B = np.maximum(0, eff_mu_B / g_B)
            
            # Mixing for numerical stability (prevents oscillations in self-consistency loop)
            n_B = mix * new_n_B + (1 - mix) * n_B
            n_F = mix * new_n_F + (1 - mix) * n_F
            
        return r, n_B, n_F

    r_vals, n_B_vals, n_F_vals = solve_densities(g_BF)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(r_vals, n_B_vals, label='Bosons $n_B(r)$', lw=2, color='blue')
    ax2.plot(r_vals, n_F_vals, label='Fermions $n_F(r)$', lw=2, linestyle='--', color='red')
    
    if g_BF == 0:
        title_state = "Non-Interacting"
    elif g_BF < 0:
        title_state = "Attractive Core"
    elif g_BF > 10:
        title_state = "Phase Separation"
    else:
        title_state = "Weak Repulsion"
        
    ax2.set_title(f'Trap Density Profiles: {title_state} (g_BF = {g_BF})')
    ax2.set_xlabel('Radius r')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

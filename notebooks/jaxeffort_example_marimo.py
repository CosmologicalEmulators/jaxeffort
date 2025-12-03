import marimo

__generated_with = "0.10.16"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""
        # jaxeffort introduction

        In this notebook we will explain how to use some capabilities from `jaxeffort` and we will show a comparison with the original `pybird` implementation. In this tutorial you will learn how to instantiate some trained emulators, run them and get jacobians as evaluated from AD systems.

        **Important**: This notebook uses jaxeffort version 0.2.2 or later.
        """
    )
    return


@app.cell
def __():
    import jaxeffort
    import numpy as np
    import jax
    import matplotlib.pyplot as plt
    return jaxeffort, np, jax, plt


@app.cell
def __(mo):
    mo.md(
        r"""
        `jaxeffort` comes batteries included with some trained emulators. They can be accessed by
        """
    )
    return


@app.cell
def __(jaxeffort):
    P0 = jaxeffort.trained_emulators["pybird_mnuw0wacdm"]["0"]
    P2 = jaxeffort.trained_emulators["pybird_mnuw0wacdm"]["2"]
    P4 = jaxeffort.trained_emulators["pybird_mnuw0wacdm"]["4"]
    return P0, P2, P4


@app.cell
def __(mo):
    mo.md(
        r"""
        You can get a description of the emulator, together with the parameters the emulator handles, by
        """
    )
    return


@app.cell
def __(P0):
    emulator_description = P0.P11.emulator_description
    return (emulator_description,)


@app.cell
def __(mo, emulator_description):
    mo.md(
        f"""
        **Emulator Description:**
        - Author: {emulator_description['author']}
        - Parameters: {emulator_description['parameters']}
        - Details: {emulator_description['miscellanea']}
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        You can see their input parameter ranges by (they are in the same order as in the above description)
        """
    )
    return


@app.cell
def __(P0):
    input_ranges = P0.P11.in_MinMax
    return (input_ranges,)


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, let us come to the usage of `jaxeffort`. First, let us define some input cosmological parameters (we define here using a dictionary just because it is convenient later for the comparison with `pybird`, but we put them into an array before feeding them to `jaxeffort`).

        We also define some dummy bias parameters. Here we also show how to compute the growth factor by using our built-in ODE solver (this is fundamental for the rescaling part).
        """
    )
    return


@app.cell
def __(np, jaxeffort):
    b = np.ones(11)

    cosmo_dict = {
        "ln10As": 3.2,
        "ns": 0.86,
        "H0": 67.,
        "ombh2": 0.022,
        "omch2": 0.12,
        "Mν": 0.06,
        "w0": -1.8,
        "wa": 0.3,
        "z": 1.2,
    }

    cosmo = jaxeffort.w0waCDMCosmology(
        ln10As=cosmo_dict["ln10As"],
        ns=cosmo_dict["ns"],
        h=cosmo_dict["H0"]/100,
        omega_b=cosmo_dict["ombh2"],
        omega_c=cosmo_dict["omch2"],
        m_nu=cosmo_dict["Mν"],
        w0=cosmo_dict["w0"],
        wa=cosmo_dict["wa"]
    )

    θ = np.array([
        cosmo_dict["z"],
        cosmo_dict["ln10As"],
        cosmo_dict["ns"],
        cosmo_dict["H0"],
        cosmo_dict["ombh2"],
        cosmo_dict["omch2"],
        cosmo_dict["Mν"],
        cosmo_dict["w0"],
        cosmo_dict["wa"]
    ])

    D, f = cosmo.D_f_z(cosmo_dict["z"])
    return b, cosmo_dict, cosmo, θ, D, f


@app.cell
def __(mo):
    mo.md(
        r"""
        Notice that we are dividing the bias parameters by `km` and `kr`, following the same convention as `pybird`.
        """
    )
    return


@app.cell
def __(b, f, P0):
    km = 0.7
    kr = 0.35
    k = P0.P11.k_grid

    # Modify bias parameters following pybird convention
    b[4] /= km**2
    b[5] /= kr**2
    b[6] /= kr**2
    b[7] = f.item()  # required to slice 0D jax arrays
    nd = 3e-4
    b[8] /= nd
    b[9] /= nd
    b[10] /= nd
    return km, kr, k, nd


@app.cell
def __(mo):
    mo.md(
        r"""
        After this prep work has been done, you just have to feed these three elements to `jaxeffort`.
        """
    )
    return


@app.cell
def __(P0, θ, b, D):
    # Compute power spectrum
    P0_result = P0.get_Pl(θ, b, D)
    return (P0_result,)


@app.cell
def __(mo):
    mo.md(
        r"""
        Ok, fine, but are these operations fast? Let's benchmark them.

        Note: In Marimo, we'll show single evaluations instead of timeit benchmarks.
        The original notebook showed:
        - Growth factor computation: ~1.5 ms
        - Emulator forward pass: ~174 µs
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Great! Now, let us show how to compute the jacobian of these calculations! In the pure functional style approach that is leveraged by `jax`, let us define an utility function and then differentiate it.
        """
    )
    return


@app.cell
def __(P0, b, D, jax, θ):
    # Define function for P0 that depends only on θ
    def P0_func(theta):
        return P0.get_Pl(theta, b, D)

    # Compute Jacobian: shape (74, 9) - 74 k-bins, 9 cosmological parameters
    jacobian_P0 = jax.jacfwd(P0_func)(θ)
    return P0_func, jacobian_P0


@app.cell
def __(mo, jacobian_P0):
    mo.md(f"**Jacobian shape:** {jacobian_P0.shape}")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Let us now plot the results of these calculations
        """
    )
    return


@app.cell
def __(P0, P2, P4, θ, b, D, k, plt):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k[:, 0], k[:, 0] * P0.get_Pl(θ, b, D), label=r"$\ell=0$")
    ax.plot(k[:, 0], k[:, 0] * P2.get_Pl(θ, b, D), label=r"$\ell=2$")
    ax.plot(k[:, 0], k[:, 0] * P4.get_Pl(θ, b, D), label=r"$\ell=4$")
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$kP_\ell(k)$')
    ax.legend()
    plt.tight_layout()
    fig_multipoles = plt.gca()
    return fig, ax, fig_multipoles


@app.cell
def __(jacobian_P0, k, plt):
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(k[:, 0], jacobian_P0[:, 4], label=r"$\omega_\mathrm{b}$")
    ax2.plot(k[:, 0], jacobian_P0[:, 5], label=r"$\omega_\mathrm{c}$")
    ax2.legend()
    ax2.set_xlabel(r'$k$')
    ax2.set_ylabel(r'$\partial P_0(k)/\partial\alpha$')
    plt.tight_layout()
    fig_jacobian = plt.gca()
    return fig2, ax2, fig_jacobian


@app.cell
def __(P0, P2, P4, θ, b, D):
    # Compute with emulator
    P0_emu = P0.get_Pl(θ, b, D)
    P2_emu = P2.get_Pl(θ, b, D)
    P4_emu = P4.get_Pl(θ, b, D)
    return P0_emu, P2_emu, P4_emu


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, let us focus on some validations. How does it compare with the original `pybird`?

        **Important**. Here everything is set to use the exact same settings used to create the training dataset. If you want something different this is fine, but you will need a different emulator. If you want to change the input cosmological parameters, just change the initial dictionary.

        **Note**: The PyBird comparison requires additional packages (classy, pybird) which need to be installed separately.
        """
    )
    return


@app.cell
def __(mo):
    # Checkbox to enable PyBird comparison (requires additional installation)
    enable_pybird = mo.ui.checkbox(label="Enable PyBird comparison (requires classy and pybird)")
    return (enable_pybird,)


@app.cell
def __(enable_pybird, mo):
    mo.md(
        f"""
        **PyBird comparison:** {'Enabled' if enable_pybird.value else 'Disabled'}

        {mo.md('Install required packages: `pip install classy git+https://github.com/pierrexyz/pybird`') if not enable_pybird.value else ''}
        """
    )
    return


@app.cell
def __(enable_pybird, cosmo_dict, np):
    # Only run PyBird if enabled
    if enable_pybird.value:
        from classy import Class
        from pybird.correlator import Correlator

        z = cosmo_dict["z"]

        # Setup CLASS parameters
        cosmo_params = {
            "output": "mPk",
            "P_k_max_h/Mpc": 20.0,
            "z_pk": "0.0,3.",
            "h": cosmo_dict["H0"] / 100,
            "omega_b": cosmo_dict["ombh2"],
            "omega_cdm": cosmo_dict["omch2"],
            "ln10^{10}A_s": cosmo_dict["ln10As"],
            "n_s": cosmo_dict["ns"],
            "tau_reio": 0.0568,
            "N_ur": 2.033,
            "N_ncdm": 1,
            "m_ncdm": cosmo_dict["Mν"],
            "use_ppf": "yes",
            "w0_fld": cosmo_dict["w0"],
            "wa_fld": cosmo_dict["wa"],
            "fluid_equation_of_state": "CLP",
            "cs2_fld": 1.,
            "Omega_Lambda": 0.,
            "Omega_scf": 0.
        }

        # Initialize CLASS and compute linear power spectrum
        M = Class()
        M.set(cosmo_params)
        M.compute()

        # Generate k values and compute linear power spectrum
        kk = 10 ** np.linspace(-5, 0, 200)
        pk_lin = [M.pk_cb(k * M.h(), z) * M.h()**3 for k in kk]

        # Get growth factors
        D1 = M.scale_independent_growth_factor(z)
        f1 = M.scale_independent_growth_factor_f(z)

        # Initialize PyBird Correlator
        N = Correlator()
        dk = 0.004
        kd = np.arange(0.005, 0.3, dk)

        # Set correlator parameters
        N.set({
            "output": "bPk",
            "multipole": 3,
            "kmax": 0.3,
            "xdata": kd,
            "km": 0.7,
            "kr": 0.35,
            "nd": 3e-4,
            "eft_basis": "eftoflss",
            "with_stoch": True,
            "with_bias": False,
            "with_resum": True
        })

        # Compute correlator
        N.compute({
            "kk": kk,
            "pk_lin": pk_lin,
            "f": f1
        })

        # Get the multipole components (matching training data)
        P11l = N.bird.P11l
        Ploopl = N.bird.Ploopl
        Pctl = N.bird.Pctl

        eft_params = {
            'b1': 1., 'b2': 1.0, 'b3': 1., 'b4': 1.,
            'cct': 1., 'cr1': 1., 'cr2': 1.,
            'ce0': 1.0, 'ce1': 1.0, 'ce2': 1.
        }

        bPk = N.get(eft_params)
        pybird_available = True
    else:
        bPk = None
        pybird_available = False

    return bPk, pybird_available


@app.cell
def __(mo, pybird_available):
    mo.md(
        f"""
        Now, let us compare the two results! We will show both the two lines in the same plot and the percentage residuals.

        {'Comparison plots will appear below.' if pybird_available else 'Enable PyBird comparison above to see comparison plots.'}
        """
    )
    return


@app.cell
def __(pybird_available, bPk, P0_emu, k, plt):
    if pybird_available and bPk is not None:
        ratio_effort_p0 = P0_emu / bPk[0, :]
        fig_p0, (ax_top_p0, ax_bot_p0) = plt.subplots(
            2, 1,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
            figsize=(10, 8)
        )

        # Top panel: power spectra
        ax_top_p0.plot(k[:, 0], k[:, 0] * bPk[0, :], label=r'$\mathrm{pybird}$')
        ax_top_p0.plot(k[:, 0], k[:, 0] * P0_emu, label=r'$\mathrm{jaxeffort}$', linestyle="--")
        ax_top_p0.set_ylabel(r'$kP_0(k)$')
        ax_top_p0.legend()

        # Bottom panel: ratio
        ax_bot_p0.plot(k[:, 0], 100 * (1 - ratio_effort_p0), color='royalblue')
        ax_bot_p0.set_xlabel(r'$k$')
        ax_bot_p0.set_ylabel(r'$\%\,\mathrm{residuals}$')

        plt.tight_layout()
        fig_comparison_p0 = plt.gca()
    else:
        fig_comparison_p0 = None

    return fig_p0, ax_top_p0, ax_bot_p0, ratio_effort_p0, fig_comparison_p0


@app.cell
def __(pybird_available, bPk, P2_emu, k, plt):
    if pybird_available and bPk is not None:
        ratio_effort_p2 = P2_emu / bPk[1, :]
        fig_p2, (ax_top_p2, ax_bot_p2) = plt.subplots(
            2, 1,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
            figsize=(10, 8)
        )

        # Top panel: power spectra
        ax_top_p2.plot(k[:, 0], k[:, 0] * bPk[1, :], label=r'$\mathrm{pybird}$')
        ax_top_p2.plot(k[:, 0], k[:, 0] * P2_emu, label=r'$\mathrm{jaxeffort}$', linestyle="--")
        ax_top_p2.set_ylabel(r'$kP_2(k)$')
        ax_top_p2.legend()

        # Bottom panel: ratio
        ax_bot_p2.plot(k[:, 0], 100 * (1 - ratio_effort_p2), color='royalblue')
        ax_bot_p2.set_xlabel(r'$k$')
        ax_bot_p2.set_ylabel(r'$\%\,\mathrm{residuals}$')

        plt.tight_layout()
        fig_comparison_p2 = plt.gca()
    else:
        fig_comparison_p2 = None

    return fig_p2, ax_top_p2, ax_bot_p2, ratio_effort_p2, fig_comparison_p2


@app.cell
def __(pybird_available, bPk, P4_emu, k, plt):
    if pybird_available and bPk is not None:
        ratio_effort_p4 = P4_emu / bPk[2, :]
        fig_p4, (ax_top_p4, ax_bot_p4) = plt.subplots(
            2, 1,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
            figsize=(10, 8)
        )

        # Top panel: power spectra
        ax_top_p4.plot(k[:, 0], k[:, 0] * bPk[2, :], label=r'$\mathrm{pybird}$')
        ax_top_p4.plot(k[:, 0], k[:, 0] * P4_emu, label=r'$\mathrm{jaxeffort}$', linestyle="--")
        ax_top_p4.set_ylabel(r'$kP_4(k)$')
        ax_top_p4.legend()

        # Bottom panel: ratio
        ax_bot_p4.plot(k[:, 0], 100 * (1 - ratio_effort_p4), color='royalblue')
        ax_bot_p4.set_xlabel(r'$k$')
        ax_bot_p4.set_ylabel(r'$\%\,\mathrm{residuals}$')

        plt.tight_layout()
        fig_comparison_p4 = plt.gca()
    else:
        fig_comparison_p4 = None

    return fig_p4, ax_top_p4, ax_bot_p4, ratio_effort_p4, fig_comparison_p4


if __name__ == "__main__":
    app.run()

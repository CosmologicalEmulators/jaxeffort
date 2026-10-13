# JAXEffort Architecture Documentation

## System Overview

JAXEffort is a JAX-based galaxy power spectrum emulator that leverages neural networks to efficiently compute cosmological power spectra with galaxy bias modeling. The system is built on top of the JAXace library and uses the JAX ecosystem for automatic differentiation and JIT compilation.

## Architecture Diagrams

### 1. High-Level System Architecture

```mermaid
graph TB
    subgraph "External Dependencies"
        JAXace[JAXace Library]
        JAX[JAX/Flax]
    end

    subgraph "JAXEffort Core"
        MLP[MLP Class<br/>Neural Network Wrapper]
        ME[MultipoleEmulators<br/>P11 + Ploop + Pct]
        MNE[MultipoleNoiseEmulator<br/>With Noise Terms]
    end

    subgraph "Data Flow"
        Input[Cosmological Parameters<br/>+ Bias Parameters]
        Output[Power Spectra<br/>P₀, P₂]
    end

    subgraph "Loading System"
        Loader[load_multipole_emulator]
        CompLoader[load_component_emulator]
        BiasLoader[load_bias_contraction]
        PreProc[load_preprocessing]
    end

    JAXace --> MLP
    JAX --> MLP
    MLP --> ME
    MLP --> MNE
    ME --> MNE

    Loader --> CompLoader
    Loader --> BiasLoader
    CompLoader --> MLP

    Input --> ME
    Input --> MNE
    ME --> Output
    MNE --> Output
```

### 2. Class Hierarchy and Relationships

```mermaid
classDiagram
    class MLP {
        +emulator: FlaxEmulator
        +k_grid: jnp.ndarray
        +in_MinMax: jnp.ndarray
        +out_MinMax: jnp.ndarray
        +postprocessing: callable
        +predict(input_vector)
        +get_component(cosmology, D)
    }

    class MultipoleEmulators {
        +P11: MLP
        +Ploop: MLP
        +Pct: MLP
        +bias_contraction: callable
        +get_Pl(cosmology, biases, D)
        +get_Pl_no_bias(cosmology, D)
    }

    class MultipoleNoiseEmulator {
        +P11: MLP
        +Ploop: MLP
        +Pct: MLP
        +bias_contraction: callable
        +get_Pl(cosmology, biases, D)
        +get_Pl_no_bias(cosmology, D)
        +get_Pl_with_stoch(cosmology, biases, D, stoch_0, stoch_2)
    }

    class FlaxEmulator {
        <<external>>
        +parameters: dict
        +apply(params, inputs)
    }

    MLP --> FlaxEmulator : wraps
    MultipoleEmulators *-- MLP : contains 3
    MultipoleNoiseEmulator *-- MLP : contains 3
    MultipoleNoiseEmulator --|> MultipoleEmulators : extends functionality
```

### 3. Data Processing Pipeline

```mermaid
graph LR
    subgraph "Input Processing"
        Cosmo[Cosmological<br/>Parameters]
        Bias[Bias<br/>Parameters]
        Growth[Growth<br/>Factor D]
    end

    subgraph "Neural Network Inference"
        Norm1[Input<br/>Normalization]
        NN[Neural Network<br/>Forward Pass]
        Norm2[Output<br/>Denormalization]
        Post[Postprocessing<br/>Function]
    end

    subgraph "Component Computation"
        P11[P₁₁<br/>Linear]
        Ploop[Ploop<br/>1-Loop]
        Pct[Pct<br/>Counter Terms]
    end

    subgraph "Bias Contraction"
        BC[Bias<br/>Contraction<br/>Matrix]
        Stack[Stack<br/>Components]
        Contract[Contract<br/>with Biases]
    end

    subgraph "Stochastic Terms"
        Stoch[Stochastic<br/>Terms<br/>ε₀, ε₁, ε₂]
        ShotNoise[Shot Noise<br/>1/n̄]
    end

    subgraph "Output"
        P0[P₀<br/>Monopole]
        P2[P₂<br/>Quadrupole]
    end

    Cosmo --> Norm1
    Bias --> BC
    Growth --> P11
    Growth --> Ploop
    Growth --> Pct

    Norm1 --> NN
    NN --> Norm2
    Norm2 --> Post

    Post --> P11
    Post --> Ploop
    Post --> Pct

    P11 --> Stack
    Ploop --> Stack
    Pct --> Stack
    Stack --> Contract
    BC --> Contract

    Contract --> P0
    Contract --> P2

    Stoch --> P0
    Stoch --> P2
```

### 4. File System Structure

```mermaid
graph TD
    subgraph "Emulator Directory Structure"
        Root[Emulator Root/]
        Root --> Dir11[11/<br/>Linear Component]
        Root --> DirLoop[loop/<br/>1-Loop Component]
        Root --> DirCt[ct/<br/>Counter Terms]
        Root --> BiasFile[biascontraction.npz<br/>or BiasContraction.npz]

        Dir11 --> Files11[nn_dict.json<br/>in_MinMax.npz<br/>out_MinMax.npz<br/>k.npz<br/>NN_params.npz]
        DirLoop --> FilesLoop[nn_dict.json<br/>in_MinMax.npz<br/>out_MinMax.npz<br/>k.npz<br/>NN_params.npz]
        DirCt --> FilesCt[nn_dict.json<br/>in_MinMax.npz<br/>out_MinMax.npz<br/>k.npz<br/>NN_params.npz]
    end
```

### 5. Computation Flow for Power Spectrum

```mermaid
sequenceDiagram
    participant User
    participant ME as MultipoleEmulators
    participant MLP as MLP Components
    participant NN as Neural Network
    participant BC as Bias Contraction

    User->>ME: get_Pl(cosmology, biases, D)

    ME->>MLP: P11.get_component(cosmology, D)
    MLP->>NN: predict(normalized_input)
    NN-->>MLP: raw_output
    MLP-->>ME: P11_component

    ME->>MLP: Ploop.get_component(cosmology, D)
    MLP->>NN: predict(normalized_input)
    NN-->>MLP: raw_output
    MLP-->>ME: Ploop_component

    ME->>MLP: Pct.get_component(cosmology, D)
    MLP->>NN: predict(normalized_input)
    NN-->>MLP: raw_output
    MLP-->>ME: Pct_component

    ME->>ME: Stack components
    ME->>BC: bias_contraction(biases, stacked_array)
    BC-->>ME: [P0, P2] multipoles

    ME-->>User: Power spectra [P0, P2]
```

### 6. Key Design Patterns

```mermaid
graph TD
    subgraph "Design Patterns"
        Wrapper[Wrapper Pattern<br/>MLP wraps FlaxEmulator]
        Factory[Factory Pattern<br/>load_multipole_emulator<br/>creates configured instances]
        Composition[Composition<br/>MultipoleEmulators<br/>composes 3 MLPs]
        Strategy[Strategy Pattern<br/>Swappable bias_contraction<br/>and postprocessing]
    end

    subgraph "JAX Integration"
        JIT[JIT Compilation<br/>@jax.jit decorators]
        Grad[Automatic Differentiation<br/>jax.grad support]
        Vmap[Vectorization<br/>jax.vmap for batches]
    end

    Wrapper --> MLP
    Factory --> load_multipole_emulator
    Composition --> MultipoleEmulators
    Strategy --> bias_contraction

    JIT --> MLP
    Grad --> MLP
    Vmap --> MultipoleEmulators
```

## Component Descriptions

### Core Emulator Classes

1. **MLP**: Wrapper around JAXace's FlaxEmulator that handles:
   - Input/output normalization
   - Neural network inference
   - Postprocessing transformations
   - Component computation with growth factors

2. **MultipoleEmulators**: Combines three MLP components:
   - P11: Linear power spectrum
   - Ploop: One-loop corrections
   - Pct: Counter terms for UV regularization
   - Applies bias contraction to get multipoles

3. **MultipoleNoiseEmulator**: Extends MultipoleEmulators with:
   - Stochastic term computation
   - Shot noise contributions
   - Combined deterministic + stochastic output

### Loading System

The loading system follows a hierarchical pattern:
- `load_multipole_emulator`: Top-level factory function
- `load_component_emulator`: Loads individual MLP components
- `load_bias_contraction`: Loads bias contraction matrices
- `load_preprocessing`: Dynamically loads postprocessing functions

### Projection Layer (`projection.py`)

Handles the mapping from theoretical power spectra to observed ones.

4. **`gausslobatto(n)`**: Dynamically computes Gauss-Lobatto-Legendre quadrature nodes and weights. A faithful port of Julia's `FastGaussQuadrature.gausslobatto`, accurate to ~1e-10 relative to the Julia reference. Results are cached via `@lru_cache` (returned arrays are read-only to protect the cache).

5. **`apply_AP`**: Applies the Alcock-Paczynski geometric distortion to a set of power spectrum multipoles using Gauss-Lobatto quadrature for the angular (μ) integration.

6. **`ChebyshevOperator` / `prepare_chebyshev_operator` / `apply_chebyshev_operator`**: Precomputes a combined linear operator `M_prime = M @ T_mat` from a projection matrix `M` and Chebyshev basis `T_mat`, enabling fast repeated applications.

7. **`APWindowChebyshevPlan` / `prepare_ap_window_chebyshev` / `apply_AP_and_window`**: A single fused plan combining the AP distortion, Gauss-Lobatto integration, and survey window convolution in one efficient pass.

### Data Flow

1. **Input**: Cosmological parameters + galaxy bias parameters
2. **Normalization**: MinMax scaling to neural network range
3. **Inference**: Forward pass through trained neural networks
4. **Denormalization**: Scale back to physical units
5. **Postprocessing**: Apply any additional transformations
6. **Bias Contraction**: Combine components with bias model
7. **AP Correction**: Remap k, μ coordinates via q_par, q_perp; integrate over μ with GL quadrature
8. **Window Convolution**: Apply survey window matrix
9. **Output**: Observed multipoles P₀, P₂, (P₄)

## Dependencies

- **JAXace**: Provides cosmology functions and neural network infrastructure
- **JAX/Flax**: Core ML framework for automatic differentiation
- **NumPy / SciPy**: Array operations, spline interpolation, and reference quadrature
- **pyartifacts**: Artifact download and cache management (mirrors Julia's `Artifacts.jl`)
- **functools.lru_cache**: In-process caching for pure functions (e.g., `gausslobatto`)

## Performance Characteristics

- **JIT Compilation**: All critical paths are JIT-compilable
- **Vectorization**: Supports batch processing via vmap
- **Gradient Support**: Fully differentiable for optimization
- **Memory Efficiency**: Operates on JAX arrays for GPU acceleration

## How to Visualize These Diagrams

To render these Mermaid diagrams:

1. **GitHub**: View this file directly on GitHub - it automatically renders Mermaid
2. **VS Code**: Install the "Markdown Preview Mermaid Support" extension
3. **Online**: Use [Mermaid Live Editor](https://mermaid.live/)
4. **Export**: Copy any diagram to Mermaid Live Editor to export as PNG/SVG

## Usage Example

```python
# Load trained emulator
emulator = load_multipole_emulator("/path/to/emulator/")

# Define cosmological parameters
cosmology = jnp.array([...])  # Cosmological parameters
biases = jnp.array([...])      # Galaxy bias parameters
D = 1.0                        # Growth factor

# Compute power spectra
P0, P2 = emulator.get_Pl(cosmology, biases, D)
```

### 7. Observational Effects / Projections

```mermaid
graph TD
    subgraph "Theoretical Outputs"
        P0_input[P₀ Monopole]
        P2_input[P₂ Quadrupole]
        P4_input[P₄ Hexadecapole]
    end

    subgraph "Alcock-Paczynski Mapping"
        Distances["Distances dA, E"]
        qp["q_perp, q_par Scalars"]
        k_mu["Coordinate Mapping k_t, μ_t"]
        Spline[Cubic / Akima Interpolation]
        GL["gausslobatto(n) nodes+weights\n(lru_cache'd)"]
        AP[Gauss-Lobatto μ Integration]
    end

    subgraph "Chebyshev Pipeline"
        ChebyOp["ChebyshevOperator\nprecomputed T-matrix"]
        APWin["APWindowChebyshevPlan\ncombined AP + window"]
    end

    subgraph "Survey Systematics"
        Window[Window Function Matrix]
        Conv[Matrix Dot Product]
    end

    subgraph "Observed Statistics"
        P_obs[Observed Power Spectrum]
    end

    Distances --> qp
    qp --> k_mu
    P0_input --> Spline
    P2_input --> Spline
    P4_input --> Spline
    k_mu --> Spline
    GL --> AP
    Spline --> AP
    AP --> Conv
    Window --> Conv
    Conv --> P_obs

    Spline --> APWin
    GL --> APWin
    ChebyOp --> APWin
    Window --> APWin
    APWin --> P_obs
```

### 8. Artifact Management System

```mermaid
graph TD
    subgraph "Artifacts.toml"
        TOML["Artifacts.toml\nURL + hash metadata"]
    end

    subgraph "pyartifacts Manager"
        Mgr[_get_artifact_manager]
        Exists[manager.exists]
        GetPath["manager.get_path\n(downloads on demand)"]
    end

    subgraph "Load on import"
        EnvVar["JAXEFFORT_NO_AUTO_DOWNLOAD\n(CI / dev override)"]
        AutoLoad["_load_emulator_set\n(auto_download=True)"]
        Fallback["Graceful fallback\n{0:None, 2:None, 4:None}"]
    end

    subgraph "trained_emulators dict"
        Dict["trained_emulators\n{model_name: {l: emulator}}"]
    end

    TOML --> Mgr
    Mgr --> Exists
    Exists -->|present| GetPath
    Exists -->|absent| GetPath
    GetPath -->|download| AutoLoad
    AutoLoad -->|ok| Dict
    AutoLoad -->|error| Fallback
    Fallback --> Dict
    EnvVar -->|set| Fallback
    EnvVar -->|unset| AutoLoad
```

Artifact downloads happen automatically at import time unless `JAXEFFORT_NO_AUTO_DOWNLOAD=1` is set.  Network errors are caught and result in `None` entries in `trained_emulators`, which cause individual tests to `pytest.skip` rather than fail.

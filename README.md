# Interval-enclosure

Rigorous interval enclosure for ODE solution using Taylor models
and different integration system

## Non-Rust dependencies

- This crate uses the [rust-matplotlib](https://github.com/Chris00/rust-matplotlib) binding for plotting, which requires:

  - A Python installation with a shared library
  - Matplotlib installed (e.g., `pip install matplotlib`)

- The crate uses [inari](https://github.com/steffahn/inari) for interval arithmetic, which depends on [MPFR](https://www.mpfr.org/) and [GMP](https://gmplib.org/).

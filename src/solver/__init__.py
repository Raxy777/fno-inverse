"""Forward physics: staggered-grid elastic FDTD and its frequency-domain reduction."""

from .fdtd_elastic import (  # noqa: F401
    ElasticFDTD2D,
    SimResult,
    dft_at_freqs,
    homogeneous_material,
    material_with_voids,
    source_spectrum,
    tone_burst,
)

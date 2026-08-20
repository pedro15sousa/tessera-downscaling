"""Convolutional VAE that turns TESSERA patches into per-station descriptors.

The downscaler conditions on a 16-dimensional vector per station. That vector
is the posterior mean of this VAE, trained to compress the 64x64x128 TESSERA
embedding patch centred on the station (:mod:`.model`, :mod:`.losses`,
:mod:`.dataset`, :mod:`.blocks`). Command-line entry points live in
``scripts/patch_encoder/``: ``train_vae.py`` fits an encoder, ``eval_vae.py``
writes the station-aligned ``eval/station_latents.npy`` consumed by
``tessera-train`` (via ``--vae-latents-path``), and ``encode_dense_grid.py``
does the same for the dense grids behind the map figures.

The encoder is frozen after training -- no gradient flows into it from the
downscaler; the latents are z-scored once and reused across every experiment.
"""

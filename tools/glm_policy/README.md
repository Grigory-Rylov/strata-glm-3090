# Expert-tier policy simulators (GLM-5.3-Flash)

Offline replays of logged routing (`strata-glm --routes`) that chose the engine's cache policies before they were
built. `GLM_DATA` points at the traces (`routes_chat_code.bin`, `routes_chat_uk.bin`, `profile_boot/`), `GLM_PROFILE`
at the boot profile. `tier_sim.py` compares RAM eviction policies (results in `results.txt`), `hybrid_sim.py` /
`hybrid_resel.py` the static + LRU VRAM split and its re-selection, `vram_belady.py` the Belady ceiling.

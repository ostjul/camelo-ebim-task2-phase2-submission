# Camelo — EBiM Task 2 real-robot submission. Station-side targets only;
# the policy server is `docker compose up policy-server` (compose.yaml).
PY ?= python3

# camelo's own Fast DDS profile (big socket buffers) rendered from the
# station's: the site profile drops the wrist cameras to 10-15 Hz.
DDS_SRC ?= /tmp/tmr_fastdds_laptop_$(shell id -u).xml
DDS_DST ?= outputs/rig/fastdds_camelo.xml
DDS_RX_MB ?= 16
DDS_TX_MB ?= 4

.PHONY: help dds-profile check-obs
help:                 ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/ —/'

dds-profile:          ## render camelo's Fast DDS profile from the station's (README: Install on the station)
	$(PY) scripts/render_dds_profile.py --src $(DDS_SRC) --dst $(DDS_DST) \
		--rx-mb $(DDS_RX_MB) --tx-mb $(DDS_TX_MB)

check-obs:            ## cameras at the wire rate and shape (README: Arms and scene)
	$(PY) scripts/check_obs.py --world real --seconds 10

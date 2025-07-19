# rubin_nights
Tools for accessing Rubin data relevant for nightly visit investigations.


To install rubin_nights within the RSP: 

```pip install --user --upgrade git+https://github.com/lsst-sims/rubin_nights.git  --no-deps```

All of the dependencies are already present in the `rubin-env-rsp` environment, and the RSP does not appreciate you trying to update these, so `--no-deps` works great.


Outside the RSP (such as in your own conda environment), you can pip install similarly for the basic requirements: 

```pip install --upgrade git+https://github.com/lsst-sims/rubin_nights.git```

But, you may also be interested in some extra dependencies linked to rubin, such as lsst-resources and rubin-scheduler + rubin-sim. Adding lsst-resources enables you to download snapshots, etc. from the LFA. Adding rubin-scheduler and rubin-sim allow you to add additional information such as predicted zeropoint and predicted slewtimes to the visit + quicklook information gathered from the ConsDb. So then the easiest way forward is:

```pip install "rubin_nights[rubin] @ git+https://github.com/lsst-sims/rubin_nights.git"```

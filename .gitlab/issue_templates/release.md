# Release checklist

Things to do for a release in the right order:
- [ ] Check the version number in [`__init__.__version__`](/energy_toolbox/__init__.py)
- [ ] Update the [`CHANGELOG`](/CHANGELOG.md) with the release date
- [ ] Check consistency of [`CHANGELOG`](/CHANGELOG.md) (no duplicates with
  previous versions, spelling etc. )
- [ ] Integrate into `develop`
- [ ] Integrate into `master`
- [ ] Create a release tag from `master`
- [ ] Update the `environement_stable.yml` file in jupyterhub-singleuser-images/jupyterhub-cerebro-latest>

/label ~"Release"

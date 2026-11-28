# Pointer packages for the names before the rename

The project was called Recallix. These are the final releases of the old PyPI names: each installs the
renamed package and keeps old imports working, with a warning. Build and publish each folder with
`uv build && uvx twine upload dist/*` once the new packages are on PyPI.

npm needs no code: `npm deprecate recallix "Renamed to recallic: npm install recallic"` and
`npm deprecate recallix-ai-sdk "Renamed to recallic-ai-sdk"`.

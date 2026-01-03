#!/bin/bash

set -eu -o errtrace

this_dir="$(cd $(dirname $0) && pwd)"
repo_root="$(cd $this_dir/../.. && pwd)"

export PYTHONPATH="$repo_root/build/tools/torch-mlir/python_packages/torch_mlir:$repo_root/projects/pt1"

echo "::group::Run TOSA LINALG e2e integration tests"
python3 -m e2e_testing.main --config=fx_importer_tosa_linalg
echo "::endgroup::"

echo "::group::Run FxImporter e2e integration tests"
python3 -m e2e_testing.main --config=fx_importer
echo "::endgroup::"

echo "::group::Run FxImporter TOSA e2e integration tests"
python3 -m e2e_testing.main --config=fx_importer_tosa
echo "::endgroup::"

{ pkgs ? import <nixpkgs> {} }:

with pkgs;

mkShell {
  packages = [
    mkl
    rustc cargo
    (pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
      numpy
      scipy
      tqdm
      pytest
      pytest-cov
      joblib
      scikit-learn
      pandas
      mpi4py
      numba

      # requirements.txt is missing these:
      requests
      pyyaml

      #ssgetpy isn't in nixpkgs
      (pkgs.python3Packages.buildPythonPackage rec {
          pname = "ssgetpy";
          version = "1.0rc2";
          pyproject = true;
          build-system = [ setuptools ];
          buildInputs = [ requests tqdm ];
          src = pkgs.python3Packages.fetchPypi {
            inherit pname version;
            sha256 = "sha256-vymOFz8VRdRRh3IgsZErLJSveTFOnEgSpTmKOTwQELU="; 
          };})
    ]))
  ];

  shellHook = ''
    unset NIX_ENFORCE_NO_NATIVE
  '';
}

{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python311
    git
    gcc
    stdenv.cc.cc.lib
  ];

  shellHook = ''
    VENV_DIR=".venv"

    if [ ! -d "$VENV_DIR" ]; then
      echo "Creating virtual environment..."
      python3 -m venv $VENV_DIR
      source $VENV_DIR/bin/activate
      echo "Upgrading pip..."
      pip install --upgrade pip

      echo "Installing PyTorch first (required for torch-scatter build)..."
      pip install torch>=2.0.0

      echo "Installing remaining requirements..."
      pip install -r requirements.txt

      echo "Setup complete! Virtual environment created at $VENV_DIR"
    else
      source $VENV_DIR/bin/activate
    fi

    echo ""
    echo "Development environment ready!"
    echo "Python: $(which python)"
    echo "Pip: $(which pip)"
    echo ""
    echo "To deactivate: deactivate"
    echo "To reactivate: nix-shell"
  '';
}

import os
import sys
import subprocess
from script_colors import SCRIPT_TAG, ERROR_TAG

class ZPolyhedratorManager:
    """Manages the cloning, building, and running of the z_polyhedrator project."""

    def __init__(self, repo_dir):
        """
        Initialize the manager with a configurable repository directory and URL.
        :param repo_dir: Directory where the repository should be cloned.
        :param repo_url: GitHub repository URL to clone.
        """
        self.Z_POLYHEDRAL_DIR = repo_dir
        self.TARGET_DIR = "target/release/z_polyhedrator"

    @staticmethod
    def _sanitize_path(path_value: str) -> str:
        """
        Best-effort sanitizer for PATH.
        Some environments accidentally pollute PATH with multi-line messages (e.g., from setvars.sh),
        which can break shell-based `export PATH="...$PATH"` constructs due to embedded quotes/newlines.
        """
        if not path_value:
            return ""
        parts = path_value.split(os.pathsep)
        keep: list[str] = []
        seen: set[str] = set()
        for part in parts:
            if not part:
                continue
            if any(ch in part for ch in ("\n", "\r", '"', "'", "`")):
                continue
            if any(ch.isspace() for ch in part):
                continue
            if part in seen:
                continue
            seen.add(part)
            keep.append(part)
        return os.pathsep.join(keep)

    def generate_uzp(self, pattern_file, input_matrix_file, output_file_name):
        """Generate the initial UZP using z_polyhedrator, ensuring correct directory execution."""

        print(f"{SCRIPT_TAG} Generating UZP with {input_matrix_file} and {pattern_file}")

        rustup_home = "/tmp/rustup"
        cargo_home = "/tmp/.cargo_pldi25_artifact"
        cargo_bin = f"{cargo_home}/bin"

        env = os.environ.copy()
        env["PATH"] = self._sanitize_path(env.get("PATH", ""))
        
        # Check for rustup in various locations, including sourcing bashrc
        rustup_found = False
        rustup_path = None
        
        # Check in custom location
        if os.path.exists(f"{cargo_bin}/rustup"):
            rustup_found = True
            rustup_path = f"{cargo_bin}/rustup"
        else:
            # Check in PATH (may need to source bashrc)
            check_cmd = 'bash -c "source ~/.bashrc 2>/dev/null; which rustup"'
            result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                rustup_found = True
                rustup_path = result.stdout.strip()
            else:
                # Check common locations
                common_paths = [
                    os.path.expanduser("~/.cargo/bin/rustup"),
                    "/usr/local/bin/rustup",
                    "/usr/bin/rustup"
                ]
                for path in common_paths:
                    if os.path.exists(path):
                        rustup_found = True
                        rustup_path = path
                        break
        
        # If rustup doesn't exist, install it first
        if not rustup_found:
            print(f"{SCRIPT_TAG} Installing rustup...")
            install_rustup_cmd = f"""
            export RUSTUP_HOME="{rustup_home}" && \
            export CARGO_HOME="{cargo_home}" && \
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
            """
            subprocess.run(install_rustup_cmd, shell=True, check=True, executable="bash", env=env)
            env["RUSTUP_HOME"] = rustup_home
            env["CARGO_HOME"] = cargo_home
            env["PATH"] = self._sanitize_path(f"{cargo_bin}{os.pathsep}{env.get('PATH', '')}")
        elif rustup_path and os.path.dirname(rustup_path) != cargo_bin:
            # Use system rustup - don't override RUSTUP_HOME/CARGO_HOME
            # Let rustup use its default locations from the environment
            rustup_dir = os.path.dirname(rustup_path)
            env["PATH"] = self._sanitize_path(f"{rustup_dir}{os.pathsep}{env.get('PATH', '')}")
        else:
            # Rustup is in custom location
            if os.path.exists(cargo_bin):
                env["PATH"] = self._sanitize_path(f"{cargo_bin}{os.pathsep}{env.get('PATH', '')}")
            env["RUSTUP_HOME"] = rustup_home
            env["CARGO_HOME"] = cargo_home
        
        rustflags = "-C opt-level=3 -C target-cpu=native"
        env_with_rustflags = env.copy()
        env_with_rustflags["RUSTFLAGS"] = rustflags

        zpoly_bin = os.path.join(self.Z_POLYHEDRAL_DIR, self.TARGET_DIR)

        try:
            subprocess.run(["rustup", "install", "1.85.0"], check=True, cwd=self.Z_POLYHEDRAL_DIR, env=env)
            subprocess.run(["cargo", "build", "--release"], check=True, cwd=self.Z_POLYHEDRAL_DIR, env=env_with_rustflags)
            subprocess.run(
                [zpoly_bin, "search", pattern_file, input_matrix_file, "-w", output_file_name],
                check=True,
                cwd=self.Z_POLYHEDRAL_DIR,
                env=env,
            )
            print(f"{SCRIPT_TAG} UZP generation completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"{ERROR_TAG} UZP generation failed: {e}")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python z_polyhedrator_manager.py <input_matrix_file> <output_directory>")
        sys.exit(1)

    # Get the directory where this script is located and calculate paths relative to it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Script is in uzp-artifact/lib/scripts/, so go up two levels to get uzp-artifact/
    uzp_artifact_dir = os.path.dirname(os.path.dirname(script_dir))
    repo_directory = os.path.join(uzp_artifact_dir, "z_polyhedrator")
    pattern_file = os.path.join(repo_directory, "data", "patterns.txt")

    manager = ZPolyhedratorManager(repo_directory)

    input_matrix_file = os.path.abspath(sys.argv[1])
    output_directory = os.path.abspath(sys.argv[2])
    output_file_name = os.path.join(output_directory, os.path.basename(input_matrix_file).replace('.mtx', ''))
    manager.generate_uzp(pattern_file, input_matrix_file, output_file_name)

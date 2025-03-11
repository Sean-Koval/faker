import os
import shutil
import logging
import datetime
import stat
from pathlib import Path
import subprocess

def cleanup_output_directory(output_dir: str = "./output"):
    """Deletes all files and subdirectories in the output directory except the latest run.

    Args:
        output_dir: The path to the output directory.
    """

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    output_path = Path(output_dir)

    # Check if the output directory exists
    if not output_path.exists():
        logger.warning(f"Output directory not found: {output_dir}")
        return

    # Collect all subdirectories and their modification times
    runs = []
    for entry in output_path.iterdir():
        if entry.is_dir():
            runs.append((entry, entry.stat().st_mtime))

    # Check if any runs were found
    if not runs:
        logger.info(f"No runs found in output directory: {output_dir}")
        return

    # Sort runs by modification time (latest first)
    runs.sort(key=lambda x: x[1], reverse=True)

    # Keep the latest run
    latest_run_path = runs[0][0]
    logger.info(f"Keeping latest run: {latest_run_path}")

    # Delete the rest
    for run_path, _ in runs[1:]:
        try:
            logger.info(f"Deleting: {run_path}")

            # Use subprocess to run 'chmod' and 'rm' commands
            for root, dirs, files in os.walk(run_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Use subprocess to change file permissions (recursively)
                    subprocess.run(['chmod', '-R', 'u+rw', root], check=True, capture_output=True, text=True)
                    break #break so that the command will only run once per directory.

                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    # Use subprocess to change directory permissions
                    subprocess.run(['chmod', '-R', 'u+rwx', root], check=True, capture_output=True, text=True)
                    break #break so that the command will only run once per directory.
            
            shutil.rmtree(run_path)

        except subprocess.CalledProcessError as e:
                logger.error(f"Error changing permissions or deleting {run_path}: {e}")
                logger.error(f"Stdout: {e.stdout}")
                logger.error(f"Stderr: {e.stderr}")
        except OSError as e:
            logger.error(f"Error deleting {run_path}: {e}")

    # Delete other files within the output directory
    for entry in output_path.iterdir():
        if entry != latest_run_path and entry.is_file():
            try:
                logger.info(f"Deleting file: {entry}")
                # Use subprocess to change file permissions
                subprocess.run(['chmod', 'u+rw', entry], check=True, capture_output=True, text=True)
                os.remove(entry)
            except subprocess.CalledProcessError as e:
                logger.error(f"Error changing permissions or deleting file {entry}: {e}")
                logger.error(f"Stdout: {e.stdout}")
                logger.error(f"Stderr: {e.stderr}")
            except OSError as e:
                logger.error(f"Error deleting file {entry}: {e}")

if __name__ == "__main__":
    cleanup_output_directory()

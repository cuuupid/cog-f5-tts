import os
import subprocess
import time
import yaml
from tqdm import tqdm
from urllib.parse import urlparse
import fnmatch

SIZE_THRESHOLD = 50  # MB
CACHE_URI = "gs://replicate-weights-wqzt/p_cd56ef2b0b97f4ad5a4c34a2beaf6d302c00d59d28bb5afa5568bbf8fe2d00e7/f5-tts/"
CDN = "https://weights.replicate.delivery/wqzt/"

def parse_dockerignore(fileobj):
    return [line.strip() for line in fileobj if line.strip() and not line.startswith('#')]

def should_ignore(file_path, dockerignore_patterns):
    # Ensure the file_path is relative to the current directory
    rel_path = os.path.normpath(file_path)
    if rel_path.startswith(os.path.sep):
        rel_path = rel_path[1:]

    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in dockerignore_patterns)

def add_to_dockerignore(files):
    with open('.dockerignore', 'a') as f:
        for file in files:
            f.write(f"\n{file}")

def make_manifest(manifest_filename: str = 'manifest.pget'):
    large_files = []

    # Load .dockerignore patterns
    dockerignore_patterns = []
    if os.path.exists('.dockerignore'):
        with open('.dockerignore', 'r') as f:
            dockerignore_patterns = parse_dockerignore(f)

    # Step 1: Find all files larger than SIZE_THRESHOLD
    for root, dirs, files in os.walk('.', topdown=True):
        # Modify dirs in-place to exclude ignored directories
        dirs[:] = [d for d in dirs if not should_ignore(os.path.relpath(os.path.join(root, d), '.'), dockerignore_patterns)]

        for file in files:
            filepath = os.path.join(root, file)
            rel_filepath = os.path.relpath(filepath, '.')
            if not should_ignore(rel_filepath, dockerignore_patterns):
                try:
                    if os.path.getsize(filepath) > SIZE_THRESHOLD * 1024 * 1024:
                        large_files.append((filepath, os.path.getsize(filepath)))
                except OSError as e:
                    print(f"Error accessing {filepath}: {e}")

    # Step 2: List relative filepaths and their sizes
    print("Large files found:")
    for filepath, size in large_files:
        print(f"{filepath}: {size / (1024 * 1024):.2f} MB")

    # Step 3: Confirm with user
    user_input = input("Please confirm you would like to cache these [Y/n]: ").strip().lower()
    if user_input == 'n':
        print("Ok, I won't generate a manifest at this time.")
        return

    # Step 4: Copy files to cache
    if CACHE_URI.startswith('s3://'):
        cp_command = ['aws', 's3', 'cp']
    elif CACHE_URI.startswith('gs://'):
        cp_command = ['gcloud', 'storage', 'cp']
    else:
        raise ValueError("Invalid CACHE_URI. Must start with 's3://' or 'gs://'")

    for filepath, _ in tqdm(large_files, desc="Copying files to cache"):
        dest_path = os.path.join(CACHE_URI, filepath.lstrip('./'))
        subprocess.run(cp_command + [filepath, dest_path], check=True)

    # Step 5: Generate manifest file
    with open(manifest_filename, 'w') as f:
        for filepath, _ in large_files:
            if CDN:
                parsed_uri = urlparse(CACHE_URI)
                path = parsed_uri.path.strip('/')
                url = f"{CDN.rstrip('/')}/{path}/{filepath.lstrip('./')}"
            elif CACHE_URI.startswith('s3://'):
                bucket, path = CACHE_URI[5:].split('/', 1)
                url = f"https://{bucket}.s3.amazonaws.com/{path}/{filepath.lstrip('./')}"
            else:  # gs://
                bucket, path = CACHE_URI[5:].split('/', 1)
                url = f"https://storage.googleapis.com/{bucket}/{path}/{filepath.lstrip('./')}"
            f.write(f"{url} {filepath}\n")

    # Add cached files to .dockerignore
    add_to_dockerignore([filepath for filepath, _ in large_files])
    print("Added cached files to .dockerignore")

    # Step 6: Update cog.yaml
    with open('cog.yaml', 'r') as f:
        cog_config = yaml.safe_load(f)

    build_config = cog_config.get('build', {})
    run_commands = build_config.get('run', [])

    pget_commands = [
        'curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/latest/download/pget_$(uname -s)_$(uname -m)"',
        'chmod +x /usr/local/bin/pget'
    ]

    if not all(cmd in run_commands for cmd in pget_commands):
        run_commands.extend(pget_commands)
        build_config['run'] = run_commands
        cog_config['build'] = build_config
        with open('cog.yaml', 'w') as f:
            yaml.dump(cog_config, f)
        print("Updated cog.yaml to install pget.")

    # Step 7: Update predictor file
    predict_config = cog_config.get('predict', '')
    if predict_config:
        predictor_file, predictor_class = predict_config.split(':')
        with open(predictor_file, 'r') as f:
            predictor_content = f.read()

        if 'from pget import pget_manifest' not in predictor_content:
            predictor_content = f"from pget import pget_manifest\n{predictor_content}"

        if 'def setup(self):' in predictor_content:
            predictor_content = predictor_content.replace(
                'def setup(self):',
                f"def setup(self):\n        pget_manifest('{manifest_filename}')"
            )
        else:
            predictor_content += f"\n    def setup(self):\n        pget_manifest('{manifest_filename}')\n"

        with open(predictor_file, 'w') as f:
            f.write(predictor_content)
        print(f"Updated {predictor_file} to include pget_manifest in setup method.")

def pget_manifest(manifest_filename: str='manifest.pget'):
    start = time.time()
    with open(manifest_filename, 'r') as f:
        manifest = f.read()

    to_dl = []
    # ensure directories exist
    for line in manifest.splitlines():
        _, path = line.split(" ")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            to_dl.append(line)

    # write new manifest
    with open("tmp.pget", 'w') as f:
        f.write("\n".join(to_dl))

    # download using pget
    subprocess.check_call(["pget", "multifile", "tmp.pget"])

    # log metrics
    timing = time.time() - start
    print(f"Downloaded weights in {timing} seconds")
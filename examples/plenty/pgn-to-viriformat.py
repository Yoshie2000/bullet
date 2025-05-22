
import bz2
from concurrent.futures import ThreadPoolExecutor
import os
import shutil
import subprocess
import tarfile

PGN_FOLDER = "/mnt/d/Chess Data/Selfgen/Data/PGNs"
VIRI_FOLDER = "/mnt/d/Chess Data/Selfgen/Data/Viriformat"
BF_FOLDER = "/mnt/d/Chess Data/Selfgen/Data/Bulletformat"
MIN_PGN_ID = 4389

# pgn_ids = list(map(lambda x: int(x.split(".")[0]), os.listdir(PGN_FOLDER)))
# viri_ids = list(map(lambda x: int(x.split(".")[0]), os.listdir(VIRI_FOLDER)))
# bf_ids = list(map(lambda x: int(x.split(".")[0]), filter(lambda x: x.endswith(".data"), os.listdir(BF_FOLDER))))
# print(sorted(set(pgn_ids) - set(viri_ids)))
# print(sorted(set(bf_ids) - set(viri_ids)))

os.makedirs("./downloads", exist_ok=True)

def decompress_bz2_file(testId, filename2):
    outName = filename2.split(".bz2")[0]
    in_path = f"./downloads/{testId}/{filename2}"
    out_path = f"./downloads/{testId}/{outName}"
    
    with open(out_path, 'wb') as new_file, open(in_path, 'rb') as file:
        decompressor = bz2.BZ2Decompressor()
        for data in iter(lambda: file.read(100 * 1024), b''):
            new_file.write(decompressor.decompress(data))
    os.remove(in_path)

for filename in os.listdir(PGN_FOLDER):
    if (not filename.endswith("tar")):
        continue
    testId = filename.split(".")[0]
    if os.path.exists(f"{VIRI_FOLDER}/{testId}.viri"):
        continue
    if int(testId) < MIN_PGN_ID:
        continue
    if int(testId) >= 4859 and int(testId) <= 4911: # fischer random
        continue
    
    if (not os.path.exists(f"./downloads/{testId}")):
        print(f"Extracting {testId}")

        # Extract tar file
        ar = tarfile.open(f"{PGN_FOLDER}/{filename}")
        ar.extractall(path=f"./downloads/{testId}")

    # Decompress bz2 files
    files = [
        f for f in os.listdir(f"./downloads/{testId}")
        if f.endswith("bz2")
    ]
    with ThreadPoolExecutor(max_workers=20) as executor:
        for f in files:
            executor.submit(decompress_bz2_file, testId, f)

    # Convert PGN to Viriformat
    print(f"Converting {testId} to viriformat")
    subprocess.call(f"./target/release/examples/pgn-to-viriformat ./downloads/{testId}", shell=True)

    # Combine files
    files = [
        f for f in os.listdir(f"./downloads/{testId}")
        if f.endswith("viri")
    ]
    with open(f'{VIRI_FOLDER}/{testId}.viri', 'wb') as out_file:
        for f in files:
            with open(f"./downloads/{testId}/{f}", 'rb') as f:
                out_file.write(f.read())

    # Delete temporary stuff
    for f in os.listdir(f"./downloads/{testId}"):
        os.remove(f"./downloads/{testId}/{f}")
    os.rmdir(f"./downloads/{testId}")
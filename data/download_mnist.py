import os
import requests
from tqdm import tqdm
import gzip
import hashlib

def download_file(url, filename, md5=None):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(filename, 'wb') as f, tqdm(desc=os.path.basename(filename), total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        if md5:
            with open(filename, 'rb') as f:
                if hashlib.md5(f.read()).hexdigest() != md5:
                    raise ValueError(f"MD5 mismatch for {filename}")
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False

def main():
    base_url = "https://yann.lecun.com/exdb/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    checksums = {
        "train_images": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        "train_labels": "d53e105ee54ea40749a1af06a2baa9ba",
        "test_images": "9dfb6290ceed2f380428a6c7554ae64",
        "test_labels": "ec29112dd5afa0611ce80d1b7f02629",
    }
    extract_dir = "/fast/slaing/data/vision/mnist"
    
    raw_dir = os.path.join(extract_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    success_count = 0
    for file_key, file_name in files.items():
        file_url = base_url + file_name
        compressed_file = os.path.join(raw_dir, file_name)
        extracted_file = os.path.join(raw_dir, file_name[:-3])
        
        if os.path.exists(extracted_file):
            print(f"{extracted_file} already exists. Skipping.")
            success_count += 1
            continue
        
        download_success = download_file(file_url, compressed_file, checksums.get(file_key))
        if download_success:
            with gzip.open(compressed_file, 'rb') as f_in, open(extracted_file, 'wb') as f_out:
                f_out.write(f_in.read())
            os.remove(compressed_file)
            print(f"Extracted {extracted_file}")
            success_count += 1
        else:
            print(f"Failed to download {file_name}")
    
    if success_count == len(files):
        print("MNIST dataset has been downloaded and extracted successfully.")
    else:
        print(f"Failed to download {len(files) - success_count} files.")

if __name__ == "__main__":
    main()
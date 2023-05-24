import os
import hashlib
import collections
import struct
import lzma


class TrieNode:
    def __init__(self):
        self.children = {}
        self.value = None


def lzw_encode_file(input_file_path, output_file_path):
    with open(input_file_path, 'rb') as input_file, open(output_file_path, 'wb') as output_file:
        # Initialize dictionary with individual bytes
        dictionary = {bytes([i]): i for i in range(256)}
        next_code = 256
        buffer = b""

        for byte in input_file.read():
            buffer += bytes([byte])
            if buffer not in dictionary:
                output_file.write(dictionary[buffer[:-1]].to_bytes(3, 'big'))
                if next_code < 2 ** 24 - 1:  # Limit dictionary size to prevent memory issues
                    dictionary[buffer] = next_code
                    next_code += 1
                buffer = buffer[-1:]

        if buffer:
            output_file.write(dictionary[buffer].to_bytes(3, 'big'))


def lzw_decode_file(input_file_path, output_file_path):
    with open(input_file_path, 'rb') as input_file, open(output_file_path, 'wb') as output_file:
        # Initialize dictionary with individual bytes
        dictionary = {i: bytes([i]) for i in range(256)}
        next_code = 256
        buffer = b""
        prev_code = None

        chunk = input_file.read(3)
        while chunk:
            code = int.from_bytes(chunk, 'big')
            if code in dictionary:
                entry = dictionary[code]
            elif code == next_code and prev_code is not None:
                entry = dictionary[prev_code] + dictionary[prev_code][:1]
            else:
                raise ValueError('Invalid compressed code.')

            output_file.write(entry)

            if prev_code is not None and next_code < 2 ** 24 - 1:
                dictionary[next_code] = dictionary[prev_code] + entry[:1]
                next_code += 1

            prev_code = code
            chunk = input_file.read(3)



def calculate_file_hash(file_path: str, block_size=65536) -> str:
    """Calculate the SHA256 hash of a file."""
    file_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            file_hash.update(block)
    return file_hash.hexdigest()

def test_lzw_coding():
    # input_file_path = input("Enter the input file path: ")
    input_file_path = "large_text_file.txt"
    file_name, file_extension = input_file_path.split(".")

    encoded_file_path = f'{file_name}_encoded_lzw.bin'
    decoded_file_path = f'{file_name}_decoded_lzw.{file_extension}'

    # Compress the file
    print("Start compression ...")
    print("Compressing", file_name)
    lzw_encode_file(input_file_path, encoded_file_path)
    print("Compression End ...")

    # Decompress the file
    print("Start decompression ...")
    print("Decompressing", file_name)
    lzw_decode_file(encoded_file_path, decoded_file_path)
    print("Decompression End ...")

    # Calculate file sizes
    original_file_size = os.path.getsize(input_file_path)
    encoded_file_size = os.path.getsize(encoded_file_path)
    decoded_file_size = os.path.getsize(decoded_file_path)

    # Calculate compression ratio (original size / compressed size)
    compression_ratio = original_file_size / encoded_file_size

    print(f'Original file size: {original_file_size} bytes')
    print(f'Encoded file size: {encoded_file_size} bytes')
    print(f'Decoded file size: {decoded_file_size} bytes')
    print(f'Compression ratio: {compression_ratio}')

    # Calculate and compare file hashes
    original_file_hash = calculate_file_hash(input_file_path)
    decoded_file_hash = calculate_file_hash(decoded_file_path)

    assert original_file_hash == decoded_file_hash, 'Hash of decoded file is not the same as the original'
    print('File hashes match')
    print("Decompressed file matches original")

def test_lzw_encoding_and_decoding():
    input_file = "large_text_file.txt"
    encoded_file = "test_encoded.lzw"
    decoded_file = "test_decoded.txt"

    # Encode and then decode the file
    lzw_encode_file(input_file, encoded_file)
    lzw_decode_file(encoded_file, decoded_file)

    # Calculate and compare file hashes
    original_file_hash = calculate_file_hash(input_file)
    decoded_file_hash = calculate_file_hash(decoded_file)

    assert original_file_hash == decoded_file_hash, 'Hash of decoded file is not the same as the original'
    print('File hashes match')
    print("Encoding and decoding test passed")


def create_large_text_file(file_path, size_in_mb):
    with open(file_path, 'w') as f:
        for _ in range(size_in_mb * 1024 * 1024):
            f.write('A')

if __name__ == '__main__':
    test_lzw_coding()
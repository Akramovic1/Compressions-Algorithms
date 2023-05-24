import os
import heapq
from collections import Counter
from typing import Dict, Tuple
import hashlib
import pickle

class HuffmanNode:
    def __init__(self, symbol: int, frequency: int):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.frequency < other.frequency


def build_huffman_tree(frequencies: Counter) -> HuffmanNode:
    """Builds a Huffman tree based on the given frequencies."""
    heap = [HuffmanNode(symbol, frequency) for symbol, frequency in frequencies.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left_node = heapq.heappop(heap)
        right_node = heapq.heappop(heap)
        internal_node = HuffmanNode(None, left_node.frequency + right_node.frequency)
        internal_node.left = left_node
        internal_node.right = right_node
        heapq.heappush(heap, internal_node)

    return heap[0]

def generate_huffman_codes(node: HuffmanNode, code: str ='', code_mapping: Dict[int, str] =None) -> Dict[int, str]:
    """Generates a Huffman code mapping based on the given Huffman tree."""
    if code_mapping is None:
        code_mapping = dict()

    if node.symbol is not None:
        code_mapping[node.symbol] = code
    else:
        generate_huffman_codes(node.left, code + '0', code_mapping)
        generate_huffman_codes(node.right, code + '1', code_mapping)

    return code_mapping

def huffman_encode_file(file_path: str, output_path: str):
    """Encodes a file using Huffman encoding."""
    with open(file_path, 'rb') as file:
        data = file.read()

    frequencies = Counter(data)
    huffman_tree = build_huffman_tree(frequencies)
    huffman_codes = generate_huffman_codes(huffman_tree)

    encoded_bits = ''.join(huffman_codes[symbol] for symbol in data)

    # Pad the encoded bits with zeros to make the length a multiple of 8
    num_padding_bits = 8 - len(encoded_bits) % 8
    padded_bits = encoded_bits + '0' * num_padding_bits

    # Convert the padded bits to bytes
    encoded_data = bytes(int(padded_bits[i:i + 8], 2) for i in range(0, len(padded_bits), 8))

    # Save encoded data to a file
    with open(output_path, 'wb') as file:
        pickle.dump((encoded_data, huffman_codes, num_padding_bits), file)


def huffman_decode_file(file_path: str, output_path: str):
    """Decodes a Huffman-encoded file."""
    with open(file_path, 'rb') as file:
        encoded_data, huffman_codes, num_padding_bits = pickle.load(file)

    decoded_data = []
    inv_huffman_codes = {code: symbol for symbol, code in huffman_codes.items()}
    buffer = ''

    # Convert the bytes to a string of bits
    bit_string = ''.join(format(byte, '08b') for byte in encoded_data)

    # Remove padding bits
    bit_string = bit_string[:-num_padding_bits]

    for bit in bit_string:
        buffer += bit
        if buffer in inv_huffman_codes:
            decoded_data.append(inv_huffman_codes[buffer])
            buffer = ''

    decoded_data = bytes(decoded_data)

    # Save decoded data to a file
    with open(output_path, 'wb') as file:
        file.write(decoded_data)

def calculate_file_hash(file_path: str, block_size=65536) -> str:
    """Calculate the SHA256 hash of a file."""
    file_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            file_hash.update(block)
    return file_hash.hexdigest()


def test_huffman_coding():
    # input_file_path = input("Enter the input file path: ")
    input_file_path = "large_text_file.txt"
    file_name, file_extension = input_file_path.split(".")

    encoded_file_path = f'{file_name}_encoded_huffman.bin'
    decoded_file_path = f'{file_name}_decoded_huffman.{file_extension}'

    # Encode the file
    print("Start compression ...")
    print("Compressing", file_name)
    huffman_encode_file(input_file_path, encoded_file_path)
    print("Compression End ...")

    # Decode the file
    print("Start decompression ...")
    print("Decompressing", file_name)
    huffman_decode_file(encoded_file_path, decoded_file_path)
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

    print('Huffman coding test passed')


if __name__ == '__main__':
    test_huffman_coding()
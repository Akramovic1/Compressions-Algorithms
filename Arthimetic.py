import pickle
from collections import Counter
from fractions import Fraction
import hashlib
import os

import io
from bitarray import bitarray
import pickle
from collections import defaultdict


class ArithmeticEncoder:
    def __init__(self):
        self.low = 0
        self.high = 0xFFFFFFFF
        self.bits_to_follow = 0
        self.bit_stream = bitarray()

    def encode_file(self, input_file, output_file):
        with open(input_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
            # Read the input file
            data = f_in.read()

            # Create the frequency table
            frequency_table = defaultdict(int)
            for symbol in data:
                frequency_table[symbol] += 1

            # Normalize the frequency table
            low_values, high_values, total = self.calculate_ranges(frequency_table)

            # Initialize the encoder
            self.low = 0
            self.high = 0xFFFFFFFF
            self.bits_to_follow = 0
            self.bit_stream = bitarray()

            # Encode the data
            for symbol in data:
                self.encode_symbol(symbol, low_values, high_values, total)

            # Flush the encoder
            self.flush_encoder()

            # Save the frequency table and encoded bitstream to the output file
            self.save_to_file(f_out, frequency_table)

    def encode_symbol(self, symbol, low_values, high_values, total):
        # Get the symbol range
        low = low_values[symbol]
        high = high_values[symbol]

        # Update the encoder's low and high values
        range = self.high - self.low + 1
        self.high = self.low + (range * high // total) - 1
        self.low = self.low + (range * low // total)

        # Renormalization loop
        while True:
            if self.high < 0x80000000:
                self.output_bit_plus_follow(0)
            elif self.low >= 0x80000000:
                self.output_bit_plus_follow(1)
                self.low -= 0x80000000
                self.high -= 0x80000000
            elif (self.low >= 0x40000000) and (self.high < 0xC0000000):
                self.bits_to_follow += 1
                self.low -= 0x40000000
                self.high -= 0x40000000
            else:
                break

            self.low <<= 1
            self.high = (self.high << 1) | 1

    def flush_encoder(self):
        self.bits_to_follow += 1

        if self.low < 0x80000000:
            self.output_bit_plus_follow(0)
        else:
            self.output_bit_plus_follow(1)

    def output_bit_plus_follow(self, bit):
        self.bit_stream.append(bit)

        while self.bits_to_follow > 0:
            self.bit_stream.append(1 - bit)
            self.bits_to_follow -= 1

    def calculate_ranges(self, frequency_table):
        sorted_table = sorted(frequency_table.items(), key=lambda x: x[0])

        cumulative_freq = 0
        low_values = {}
        high_values = {}
        for symbol, frequency in sorted_table:
            low = cumulative_freq
            high = cumulative_freq + frequency - 1
            cumulative_freq += frequency
            low_values[symbol] = low
            high_values[symbol] = high

        total = cumulative_freq
        return low_values, high_values, total

    def save_to_file(self, f_out, frequency_table):
        for symbol in range(256):
            frequency = frequency_table.get(symbol, 0)
            f_out.write(symbol.to_bytes(1, 'big'))
            f_out.write(frequency.to_bytes(4, 'big'))

        print("File pointer position before writing total_bits:", f_out.tell())
        f_out.write(len(self.bit_stream).to_bytes(4, 'big'))
        print("File pointer position after writing total_bits:", f_out.tell())
        f_out.write(self.bit_stream.tobytes())


class ArithmeticDecoder:
    def __init__(self):
        self.low = 0
        self.high = 0xFFFFFFFF
        self.value = 0
        self.bit_stream = bitarray()
        self.total_bits = 0

    def decode_file(self, input_file, output_file):
        with open(input_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
            # Read the frequency table and encoded bitstream from the input file
            frequency_table, bit_stream, total_bits = self.load_from_file(f_in)

            # Normalize the frequency table
            low_values, high_values, total = self.calculate_ranges(frequency_table)

            print(f"Cumulative frequency: {total}, size of original file: {os.path.getsize(input_file)}")  # Debug print

            # Initialize the decoder
            self.low = 0
            self.high = 0xFFFFFFFF
            self.value = int.from_bytes(bit_stream[:4], 'big')
            self.bit_stream = bit_stream[4:]
            self.total_bits = total_bits

            print(f"Length of bit_stream: {len(self.bit_stream)}, total_bits: {self.total_bits}")  # Debug print

            # Decode the data
            bits_read = 0
            total_bits_popped = 0  # Debug counter
            while bits_read < self.total_bits:
                symbol, bits_popped = self.decode_symbol(frequency_table, low_values, high_values, total)
                total_bits_popped += bits_popped  # Increment debug counter
                if symbol is None:
                    break
                f_out.write(symbol.to_bytes(1, 'big'))
                bits_read += 1

            print(f"Total bits popped from bit_stream: {total_bits_popped}")  # Debug print

    def decode_symbol(self, frequency_table, low_values, high_values, total):
        range = self.high - self.low + 1
        scaled_value = ((self.value - self.low + 1) * total - 1) // range

        # Find the symbol corresponding to the scaled value
        for symbol in frequency_table:
            if low_values[symbol] <= scaled_value < high_values[symbol] + 1:
                break
        else:
            return None, 0

        # Update the decoder's low and high values
        self.low = self.low + range * low_values[symbol] // total
        self.high = self.low + range * high_values[symbol] // total

        # Underflow case handling
        bits_popped = 0  # Debug counter
        while ((self.high < 0x80000000) or (self.low >= 0x80000000)) and self.bit_stream:
            if self.high < 0x80000000:
                self.low <<= 1
                self.high = (self.high << 1) | 1
                self.value = (self.value << 1) + self.bit_stream.pop(0)
                bits_popped += 1  # Increment debug counter
            elif self.low >= 0x80000000:
                self.value = ((self.value ^ 0x80000000) << 1) + self.bit_stream.pop(0)
                self.low = (self.low ^ 0x80000000) << 1
                self.high = ((self.high ^ 0x80000000) << 1) | 1

        return symbol, bits_popped

    def calculate_ranges(self, frequency_table):
        sorted_table = sorted(frequency_table.items(), key=lambda x: x[0])

        cumulative_freq = 0
        low_values = {}
        high_values = {}
        for symbol, frequency in sorted_table:
            low = cumulative_freq
            high = cumulative_freq + frequency - 1
            cumulative_freq += frequency
            low_values[symbol] = low
            high_values[symbol] = high

        total = cumulative_freq
        return low_values, high_values, total

    def load_from_file(self, f_in):
        frequency_table = {}
        for _ in range(256):
            symbol = int.from_bytes(f_in.read(1), 'big')
            frequency = int.from_bytes(f_in.read(4), 'big')
            if frequency > 0:
                frequency_table[symbol] = frequency

        print("File pointer position before reading total_bits:", f_in.tell())
        total_bits = int.from_bytes(f_in.read(4), 'big')
        print("File pointer position after reading total_bits:", f_in.tell())

        bit_stream_bytes = f_in.read()
        bit_stream = bitarray()
        bit_stream.frombytes(bit_stream_bytes)

        return frequency_table, bit_stream, total_bits

def calculate_file_hash(file_path: str, block_size=65536) -> str:
    """Calculate the SHA256 hash of a file."""
    file_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            file_hash.update(block)
    return file_hash.hexdigest()

def test_arthimetic_coding():
    # input_file_path = input("Enter the input file path: ")
    input_file_path = "FUNEX.pdf"
    file_name, file_extension = input_file_path.split(".")

    encoded_file_path = f'{file_name}_encoded_arthimetic.bin'
    decoded_file_path = f'{file_name}_decoded_arthimetic.{file_extension}'

    # Compress the file
    print("Start compression ...")
    print("Compressing", file_name)
    encoder = ArithmeticEncoder()
    encoder.encode_file(input_file_path, encoded_file_path)
    print("Compression End ...")

    # Decompress the file
    print("Start decompression ...")
    print("Decompressing", file_name)
    decoder = ArithmeticDecoder()
    decoder.decode_file(encoded_file_path, decoded_file_path)
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

def test_arthimetic_encoding_and_decoding():
    input_file = "large_text_file.txt"
    encoded_file = "test_encoded.bin"
    decoded_file = "test_decoded.txt"

    # Encode and then decode the file
    print("Start compression ...")
    encoder = ArithmeticEncoder()
    encoder.encode_file(input_file, encoded_file)

    print("Start decompression ...")
    decoder = ArithmeticDecoder()
    decoder.decode_file(encoded_file, decoded_file)

    # Calculate and compare file hashes
    original_file_hash = calculate_file_hash(input_file)
    decoded_file_hash = calculate_file_hash(decoded_file)

    assert original_file_hash == decoded_file_hash, 'Hash of decoded file is not the same as the original'
    print('File hashes match')
    print("Encoding and decoding test passed")


def create_large_text_file(file_path, size_in_mb):
    with open(file_path, 'w') as f:
        for _ in range(size_in_mb * 1024 * 1024):
            f.write('Ahmed ')


if __name__ == '__main__':
    create_large_text_file('large_text_file.txt', 1)
    test_arthimetic_encoding_and_decoding()
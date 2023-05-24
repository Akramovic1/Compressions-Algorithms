import hashlib
import os
from fractions import Fraction
from tqdm import tqdm
import math
from bisect import bisect_right
from tqdm import trange
from hashlib import md5
from termcolor import cprint

class ArithmeticEncoder:
    def __init__(self, content):
        self.content = content

    def get_symbol_probability(self, symbol):
        content_len = len(self.content)
        symbol_count = self.content.count(symbol)
        symbol_probability = Fraction(symbol_count, content_len)
        return symbol_probability

    def get_symbols_probabilities(self):
        symbols = [*set(self.content)]
        pairs = [[s, self.get_symbol_probability(s)] for s in symbols]
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        return pairs

    @staticmethod
    def get_floor_quantized_probability(probability):
        floor_exponent = abs(math.floor(math.log2(probability)))
        return Fraction(1, 2**floor_exponent)

    def get_floor_quantized_probabilities(self, symbol_probabilities):
        pairs = [[s, self.get_floor_quantized_probability(p)] for s, p in symbol_probabilities]
        return pairs

    @staticmethod
    def get_probabilities_sum(symbol_probabilities):
        return sum([x[1] for x in symbol_probabilities])

    @staticmethod
    def optimize_symbols_probabilities(symbol_probabilities, stock_floor):
        for i in range(len(symbol_probabilities)):
            probability = symbol_probabilities[i][1]
            if probability <= stock_floor:
                symbol_probabilities[i][1] *= 2
                break

    def get_optimized_probabilities(self, symbol_probabilities):
        probabilities_sum = self.get_probabilities_sum(symbol_probabilities)

        while probabilities_sum < 1.0:
            stock = Fraction(1, 1) - probabilities_sum
            stock_floor = self.get_floor_quantized_probability(stock)

            self.optimize_symbols_probabilities(symbol_probabilities, stock_floor)
            probabilities_sum = self.get_probabilities_sum(symbol_probabilities)

        return symbol_probabilities

    def get_optimal_quantized_probabilities(self):
        symbol_probabilities = self.get_symbols_probabilities()
        symbol_probabilities = self.get_floor_quantized_probabilities(symbol_probabilities)
        symbol_probabilities = self.get_optimized_probabilities(symbol_probabilities)

        return symbol_probabilities

    def get_symbols_dict(self):
        pairs = self.get_optimal_quantized_probabilities()

        recent_range_stop = Fraction(0, 1)
        symbols_dict = {}

        for symbol, probability in pairs:
            symbol_range_start = recent_range_stop
            symbol_range_delta = probability

            symbols_dict[symbol] = (symbol_range_start, symbol_range_delta)
            recent_range_stop = symbol_range_start + symbol_range_delta

        return symbols_dict

    def encode(self):
        content_md5 = md5(self.content).hexdigest()
        cprint(f'Input file MD5 sum: {content_md5}', 'yellow')
        symbols_dict = self.get_symbols_dict()

        current_range_start = Fraction(0, 1)
        current_range_delta = Fraction(1, 1)

        for c in tqdm(self.content, desc='Encoding'):
            symbol_range_start, symbol_range_delta = symbols_dict[c]

            current_range_start += (current_range_delta * symbol_range_start)
            current_range_delta *= symbol_range_delta

        return current_range_start, len(self.content), symbols_dict


class ArithmeticalDecoder:
    def __init__(self, content_fraction, content_length, symbol_dict):
        self.content_fraction = content_fraction
        self.content_length = content_length
        self.symbols_dict = symbol_dict
        self.symbols = tuple(symbol_dict.keys())
        self.ranges_starts = tuple([x[0] for x in symbol_dict.values()])

    def get_new_symbol(self):
        symbol_index = bisect_right(self.ranges_starts, self.content_fraction) - 1
        symbol = self.symbols[symbol_index]
        symbol_range_start, symbol_range_delta = self.symbols_dict[symbol]

        self.content_fraction = (self.content_fraction - symbol_range_start) / symbol_range_delta
        return symbol

    def decode(self):
        content = bytearray()

        for _ in trange(self.content_length, desc='Decoding'):
            new_symbol = self.get_new_symbol()
            content.append(new_symbol)

        content_md5 = md5(content).hexdigest()
        cprint(f'Output file MD5 sum: {content_md5}', 'yellow')

        return content

class FileWriter:
    def __init__(self, filename):
        self.file = open(filename, 'wb+')

    @staticmethod
    def _get_normalized_fraction(fraction, precision):
        max_number = 2 ** (8 * precision)
        return int(fraction * max_number)

    @staticmethod
    def _get_precision(number):
        return math.ceil(math.log2(number + 1) / 8)

    def _write_int(self, number_int, precision):
        number_bytes = number_int.to_bytes(precision, byteorder='big')
        self.file.write(number_bytes)

    def _write_symbols(self, symbols_ranges):
        symbol_count = len(symbols_ranges)
        self._write_int(symbol_count - 1, 1)

        for symbol in symbols_ranges.keys():
            self._write_int(symbol, 1)

    def _write_ranges_start(self, symbols_dict):
        ranges_start = [x[0] for x in symbols_dict.values()]
        symbol_range_precision = self._get_precision((1 - ranges_start[-1]).denominator)
        self._write_int(symbol_range_precision, 1)

        for range_start in ranges_start:
            range_start_normalized = self._get_normalized_fraction(range_start, symbol_range_precision)
            self._write_int(range_start_normalized, symbol_range_precision)

    def _write_content_length(self, content_length):
        content_length_precision = self._get_precision(content_length)
        self._write_int(content_length_precision, 1)
        self._write_int(content_length, content_length_precision)

    def _write_content_fraction(self, content_fraction):
        numerator = content_fraction.numerator
        numerator_precision = self._get_precision(numerator) + 1
        numerator_precision_precision = self._get_precision(numerator_precision)
        numerator_normalized = self._get_normalized_fraction(content_fraction, numerator_precision)

        self._write_int(numerator_precision_precision, 1)
        self._write_int(numerator_precision, numerator_precision_precision)
        self._write_int(numerator_normalized, numerator_precision)

    def write(self, content_fraction, content_length, symbols_dict):
        self._write_symbols(symbols_dict)
        self._write_ranges_start(symbols_dict)
        self._write_content_length(content_length)
        self._write_content_fraction(content_fraction)
        self.file.close()

class FileReader:
    def __init__(self, filename):
        self.file = open(filename, 'rb')

    def _read_int(self, precision):
        return int.from_bytes(self.file.read(precision), byteorder='big')

    def _read_symbols(self):
        symbol_count = self._read_int(1) + 1
        symbols = [self._read_int(1) for _ in range(symbol_count)]
        return symbols

    def _get_ranges_starts(self, symbol_count):
        precision = self._read_int(1)
        denominator = 2 ** (8 * precision)

        numerators = [self._read_int(precision) for _ in range(symbol_count)]
        ranges_starts = [Fraction(numerator, denominator) for numerator in numerators]
        return ranges_starts

    def _read_content_length(self):
        content_length_precision = self._read_int(1)
        content_length = self._read_int(content_length_precision)
        return content_length

    def _read_content_fraction(self):
        numerator_precision_precision = self._read_int(1)
        numerator_precision = self._read_int(numerator_precision_precision)
        numerator_normalized = self._read_int(numerator_precision)

        content_fraction = Fraction(numerator_normalized, 2 ** (8 * numerator_precision))
        return content_fraction

    @staticmethod
    def _get_symbols_dict(symbols, ranges_starts):
        ranges_thresholds = ranges_starts + [Fraction(1, 1)]
        symbols_dict = {}

        for i in range(len(symbols)):
            symbol_range_start = ranges_thresholds[i]
            symbol_range_delta = ranges_thresholds[i + 1] - ranges_thresholds[i]
            symbols_dict[symbols[i]] = (symbol_range_start, symbol_range_delta)

        return symbols_dict

    def read(self):
        symbols = self._read_symbols()
        ranges_starts = self._get_ranges_starts(symbol_count=len(symbols))
        symbols_dict = self._get_symbols_dict(symbols, ranges_starts)
        content_length = self._read_content_length()
        content_fraction = self._read_content_fraction()
        self.file.close()

        return content_fraction, content_length, symbols_dict

def compress(file_in, file_out):
    cprint(f'Input file size: {os.stat(file_in).st_size}', 'magenta')
    with open(file_in, 'rb') as f:
        content = f.read()

    ae = ArithmeticEncoder(content)
    content_fraction, length, symbols_dict = ae.encode()

    fw = FileWriter(file_out)
    fw.write(content_fraction, length, symbols_dict)
    cprint(f'Output file size: {os.stat(file_out).st_size}', 'green')

def decompress(file_in, file_out):
    cprint(f'Input file size: {os.stat(file_in).st_size}', 'cyan')
    fr = FileReader(file_in)
    content_fraction, length, symbols_dict = fr.read()

    ad = ArithmeticalDecoder(content_fraction, length, symbols_dict)
    decoded_content = ad.decode()

    with open(file_out, 'wb') as f:
        f.write(decoded_content)
    cprint(f'Output file size: {os.stat(file_out).st_size}', 'blue')

def calculate_file_hash(file_path: str, block_size=65536) -> str:
    """Calculate the SHA256 hash of a file."""
    file_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            file_hash.update(block)
    return file_hash.hexdigest()

def test_arthimetic_coding():
    # input_file_path = input("Enter the input file path: ")
    input_file_path = "large_text_file.txt"
    file_name, file_extension = input_file_path.split(".")

    encoded_file_path = f'{file_name}_encoded_huffman.bin'
    decoded_file_path = f'{file_name}_decoded_huffman.{file_extension}'

    # Encode the file
    print("Start compression ...")
    print("Compressing", file_name)
    compress(input_file_path, encoded_file_path)
    print("Compression End ...")

    # Decode the file
    print("Start decompression ...")
    print("Decompressing", file_name)
    decompress(encoded_file_path, decoded_file_path)
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

    print('Arithmetic coding test passed')

if __name__ == '__main__':
    test_arthimetic_coding()
import numpy as np
import cv2
import math
import streamlink
import time
from numba import jit
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.utils import Prehashed

def getImage(url):
    # get the stream with the best quality
    stream = streamlink.streams(url)
    stream_url = stream["best"].url

    # use stream's url to get frame
    streamCap = cv2.VideoCapture(stream_url)
    success, image = streamCap.read()

    # save the frame
    if success:
        cv2.imwrite("zdj.png", image)
        print("Image is saved successfully.")
    else:
        print("Image is not saved successfully.")

    streamCap.release()

    # image scaling and saving
    image = imageCorrection(image)
    return image

def imageCorrection(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def entropy(input, base):
    value, counts = np.unique(input, return_counts=True)
    len = input.size
    output = 0.0
    for x in counts:
        output += (x / len) * math.log((1 / (x / len)), base)
    return output

@jit(nopython=True)
def logisticmap(la: float, x0: float, length: int, cut: int):
    logmap = np.empty(length + cut, dtype=float)
    logmap[0] = x0
    for x in range(1, (length - 1) + 250):
        logmap[x] = la * logmap[x - 1] * (1 - logmap[x - 1])
    return logmap[cut:]

@jit(nopython=True)
def miximage(sequence, image):
    indices = np.argsort(sequence)
    new_image = np.empty_like(image)
    size = new_image.shape[0]
    for index in np.ndindex(image.shape[0], image.shape[1]):
        new_image[index[0]][index[1]] = image[(indices[(index[0] * size) + index[1] + 1] % size)][(indices[(index[0] * size) + index[1]] % size)]
    return new_image

@jit(nopython=True)
def threshold(input, c):
    output = np.where(input > c, 1, 0)
    return output

@jit(nopython=True)
def confuse(input, sequence):
    for x in range(0, input.size - 1):
        input[x] = (input[x] + sequence[x]) % 2
    return input

def split_planes(input):
    a = np.unpackbits(input)
    p_1 = np.empty(int(a.size / 8), dtype=np.uint8)
    p_2 = np.empty(int(a.size / 8), dtype=np.uint8)
    p_3 = np.empty(int(a.size / 8), dtype=np.uint8)
    p_4 = np.empty(int(a.size / 8), dtype=np.uint8)
    p_5 = np.empty(int(a.size / 8), dtype=np.uint8)
    p_6 = np.empty(int(a.size / 8), dtype=np.uint8)
    p_7 = np.empty(int(a.size / 8), dtype=np.uint8)
    p_8 = np.empty(int(a.size / 8), dtype=np.uint8)
    i = 0
    for index in range(0, a.size, 8):
        p_1[i] = a[index]
        p_2[i] = a[index + 1]
        p_3[i] = a[index + 2]
        p_4[i] = a[index + 3]
        p_5[i] = a[index + 4]
        p_6[i] = a[index + 5]
        p_7[i] = a[index + 6]
        p_8[i] = a[index + 7]
        i += 1
    return p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8

def merge_planes(*args):
    num_planes = len(args)
    if num_planes != 8:
        raise ValueError("merge_planes requires exactly 8 input arrays.")

    lengths = [len(arr) for arr in args]
    if len(set(lengths)) != 1:
        raise ValueError("All input arrays must have the same length.")

    result_size = lengths[0] * 8
    result = np.empty(result_size, dtype=np.uint8)

    for i in range(lengths[0]):
        for j in range(8):
            result[i * 8 + j] = args[j][i]

    reconstructed = np.packbits(result)
    return reconstructed

def post_process(obraz):
    data = np.array(obraz, dtype=np.uint8)
    shape = data.shape
    x_0 = logisticmap(4, 0.361, shape[0] * shape[1], 250)
    x_1 = logisticmap(4, 0.362, shape[0] * shape[1], 250)
    x_2 = logisticmap(4, 0.363, shape[0] * shape[1], 250)
    x_3 = logisticmap(4, 0.364, shape[0] * shape[1], 250)
    x_4 = logisticmap(4, 0.365, shape[0] * shape[1], 250)
    x_5 = logisticmap(4, 0.366, shape[0] * shape[1], 250)
    x_6 = logisticmap(4, 0.367, shape[0] * shape[1], 250)
    x_7 = logisticmap(4, 0.368, shape[0] * shape[1], 250)
    x_8 = logisticmap(4, 0.369, shape[0] * shape[1], 250)
    new_image = miximage(x_0, data)

    x_1 = threshold(x_1, 0.5)
    x_2 = threshold(x_2, 0.5)
    x_3 = threshold(x_3, 0.5)
    x_4 = threshold(x_4, 0.5)
    x_5 = threshold(x_5, 0.5)
    x_6 = threshold(x_6, 0.5)
    x_7 = threshold(x_7, 0.5)
    x_8 = threshold(x_8, 0.5)

    p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8 = split_planes(new_image)

    p_1 = confuse(p_1, x_1)
    p_2 = confuse(p_2, x_2)
    p_3 = confuse(p_3, x_3)
    p_4 = confuse(p_4, x_4)
    p_5 = confuse(p_5, x_5)
    p_6 = confuse(p_6, x_6)
    p_7 = confuse(p_7, x_7)
    p_8 = confuse(p_8, x_8)

    res = merge_planes(p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8)
    return res, obraz.flatten()

def create_random_bits(url, bitsamount, original_bin, random_bin):
    byteamount = int(np.ceil(bitsamount / 8))
    output_ran = np.empty(0, dtype=np.uint8)
    output_original = np.empty(0, dtype=np.uint8)
    while output_ran.size < byteamount:
        time_start = time.time()
        image = getImage(url)
        res, obraz = post_process(image)
        time_end = time.time()
        time_duration = time_end - time_start
        print("czas procesowania jednego obrazu: ", time_duration)
        print("ilosc bitow: ", res.size * 8)
        print("bity na sekunde: ", (res.size * 8) / time_duration)
        print("created_random_bits")
        output_ran = np.append(output_ran, res)
        output_original = np.append(output_original, res)
    print("creation successful, saving...")
    output_original = output_original[0:byteamount]
    output_ran = output_ran[0:byteamount]
    with open(original_bin, "wb") as binary_file:
        binary_file.write(output_original.tobytes())

    with open(random_bin, "wb") as binary_file:
        binary_file.write(output_ran.tobytes())

# Funkcja do generowania kluczy, podpisywania wiadomości oraz przeprowadzania testów
def generate_keys_and_test_signatures():
    # Generowanie klucza prywatnego i publicznego
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    public_key = private_key.public_key()

    # Serializacja i wyświetlenie kluczy
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    print("Private Key:")
    print(private_pem.decode())
    print("Public Key:")
    print(public_pem.decode())

    # Funkcja do podpisywania wiadomości
    def sign_message(private_key, message):
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature

    # Funkcja do weryfikacji podpisu
    def verify_signature(public_key, message, signature):
        try:
            public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            print(f"Verification error: {e}")
            return False

    # Tworzenie podpisu dla wiadomości
    message = b"Test message"
    signature = sign_message(private_key, message)
    print("\nSignature:")
    print(signature)

    # Test 1: Poprawna wiadomość i klucz
    is_valid = verify_signature(public_key, message, signature)
    print("\nTest 1 - Correct key and message:", is_valid)

    # Generowanie nowego klucza prywatnego (dla nieprawidłowego klucza)
    wrong_private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    wrong_public_key = wrong_private_key.public_key()

    # Test 2: Nieprawidłowy klucz publiczny
    is_valid_wrong_key = verify_signature(wrong_public_key, message, signature)
    print("Test 2 - Incorrect key:", is_valid_wrong_key)

    # Test 3: Zmodyfikowana wiadomość
    modified_message = b"Modified message"
    is_valid_modified_message = verify_signature(public_key, modified_message, signature)
    print("Test 3 - Modified message:", is_valid_modified_message)

def main():
    # Wywołanie funkcji create_random_bits
    create_random_bits("https://www.youtube.com/watch?v=mDDWtALSOw4", 30000, "original.bin", "random.bin")
    
    # Generowanie kluczy, podpisywanie wiadomości oraz przeprowadzanie testów
    generate_keys_and_test_signatures()

if __name__ == "__main__":
    main()


# RSA_Test.py

from RSA_Functions import montgomery_powmod

def test_single_rsa():
    n = 3233
    e = 17
    d = 2753
    message = 123

    cipher = montgomery_powmod(message, e, n)
    decrypted = montgomery_powmod(cipher, d, n)

    print(f"Message:    {message}")
    print(f"Encrypted:  {cipher}")
    print(f"Decrypted:  {decrypted}")
if __name__ == "__main__":
    test_single_rsa()

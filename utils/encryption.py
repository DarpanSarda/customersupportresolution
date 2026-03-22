"""
Encryption/Decryption utilities for sensitive data.

Uses AES-256-GCM encryption with PBKDF2 key derivation.
Compatible with the Node.js encryption system.
"""

import os
import base64
import hashlib
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Get encryption key from environment
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
ALGORITHM = "aes-256-gcm"
ITERATIONS = 100000
KEY_LENGTH = 32  # 256 bits for AES-256
IV_LENGTH = 16  # Node.js crypto uses 16 bytes for IV
TAG_LENGTH = 16  # Auth tag is always 16 bytes for GCM


def _get_encryption_key() -> bytes:
    """
    Derive encryption key from the base key using PBKDF2.

    Compatible with Node.js encryption: uses SHA-256 hash of 'helpgenie-salt' as salt.

    Returns:
        32-byte encryption key
    """
    if not ENCRYPTION_KEY:
        raise ValueError("ENCRYPTION_KEY environment variable is not set")

    # Use the key directly as UTF-8 bytes (matches Node.js behavior)
    base_key = ENCRYPTION_KEY.encode('utf-8')

    # Generate salt using SHA-256 hash of 'helpgenie-salt' (matches Node.js)
    salt = hashlib.sha256('helpgenie-salt'.encode()).digest()

    # Use PBKDF2HMAC to derive the actual encryption key
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_LENGTH,
        salt=salt,
        iterations=ITERATIONS,
        backend=default_backend()
    )

    return kdf.derive(base_key)


def decrypt(encrypted_data: str, iv_data: str, tag_data: str) -> str:
    """
    Decrypt encrypted data using AES-256-GCM.

    Compatible with the Node.js encryption system.
    Handles both hex and base64 formats.

    Args:
        encrypted_data: Encrypted data (hex or base64 format)
        iv_data: Initialization vector (hex or base64 format)
        tag_data: Authentication tag (hex or base64 format)

    Returns:
        Decrypted plaintext string

    Raises:
        ValueError: If decryption fails
    """
    try:
        print(f"[DECRYPT] Input lengths - encrypted: {len(encrypted_data)}, iv: {len(iv_data)}, tag: {len(tag_data)}")
        print(f"[DECRYPT] Input samples - encrypted: {encrypted_data[:50]}..., iv: {iv_data[:50]}..., tag: {tag_data[:50]}...")

        # Get the encryption key
        key = _get_encryption_key()
        print(f"[DECRYPT] Key derived successfully, length: {len(key)} bytes")

        # Try to detect and convert format
        encrypted = _decode_data(encrypted_data)
        iv = _decode_data(iv_data)
        tag = _decode_data(tag_data)

        print(f"[DECRYPT] Decoded lengths - encrypted: {len(encrypted)}, IV: {len(iv)}, Tag: {len(tag)}")
        print(f"[DECRYPT] IV (hex): {iv.hex()}")
        print(f"[DECRYPT] Tag (hex): {tag.hex()}")
        print(f"[DECRYPT] Encrypted (first 50 hex): {encrypted[:50].hex()}...")

        # AES-GCM combines the tag with the ciphertext
        # In GCM, the tag is appended to the ciphertext
        ciphertext_with_tag = encrypted + tag

        print(f"[DECRYPT] Combined length: {len(ciphertext_with_tag)}")

        # Create AES-GCM cipher and decrypt
        aesgcm = AESGCM(key)
        decrypted = aesgcm.decrypt(iv, ciphertext_with_tag, None)

        print(f"[DECRYPT] Decryption successful!")

        return decrypted.decode('utf-8')

    except Exception as e:
        print(f"[DECRYPT] Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"Decryption failed: {str(e)}")


def _decode_data(data: str) -> bytes:
    """
    Decode data from hex or base64 format.

    Args:
        data: String data in hex or base64 format

    Returns:
        Decoded bytes

    Raises:
        ValueError: If format is invalid
    """
    # Try hex first
    try:
        return bytes.fromhex(data)
    except ValueError:
        pass

    # Try base64
    try:
        return base64.b64decode(data)
    except Exception as e:
        raise ValueError(f"Unable to decode data as hex or base64: {str(e)}")


def encrypt(plaintext: str) -> tuple[str, str, str]:
    """
    Encrypt plaintext using AES-256-GCM.

    Compatible with the Node.js encryption system.

    Args:
        plaintext: String to encrypt

    Returns:
        Tuple of (encrypted_data, iv_data, tag_data) in hex format
    """
    try:
        # Get the encryption key
        key = _get_encryption_key()

        # Generate random IV (16 bytes to match Node.js)
        iv = os.urandom(IV_LENGTH)

        # Create AES-GCM cipher and encrypt
        aesgcm = AESGCM(key)
        ciphertext_with_tag = aesgcm.encrypt(iv, plaintext.encode(), None)

        # Split ciphertext and tag (tag is last 16 bytes in GCM)
        ciphertext = ciphertext_with_tag[:-16]
        tag = ciphertext_with_tag[-16:]

        # Convert to hex
        encrypted_hex = ciphertext.hex()
        iv_hex = iv.hex()
        tag_hex = tag.hex()

        return encrypted_hex, iv_hex, tag_hex

    except Exception as e:
        raise ValueError(f"Encryption failed: {str(e)}")


# Test function to verify compatibility
def test_encryption():
    """Test encryption/decryption."""
    test_data = "test-api-key-12345"

    encrypted, iv, tag = encrypt(test_data)
    decrypted = decrypt(encrypted, iv, tag)

    assert decrypted == test_data, f"Encryption test failed: {decrypted} != {test_data}"
    print("✓ Encryption/decryption test passed")

    return True

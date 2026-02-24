"""Генератор лицензионных ключей без хранения списка ключей в приложении.

Формат: FBOT-<7|14|30>-<B5>-<B8>-<B3>-<B9>-<checksum>
- B5: 4-значный блок, кратный 5
- B8: 4-значный блок, кратный 8
- B3: 4-значный блок, кратный 3
- B9: 4-значный блок, кратный 9
- checksum: первые 2 hex символа sha256 от всей строки без checksum
"""

from __future__ import annotations

import argparse
import hashlib
import random

ALLOWED_DURATIONS = (7, 14, 30)


class LicenseKeyGenerator:
    PREFIX = "FBOT"

    @staticmethod
    def _checksum(base: str) -> str:
        return hashlib.sha256(base.encode("utf-8")).hexdigest().upper()[:2]

    @staticmethod
    def _block_multiple_of(divisor: int) -> str:
        # 4-значный диапазон: 1000..9999
        low = (1000 + divisor - 1) // divisor
        high = 9999 // divisor
        n = random.randint(low, high) * divisor
        return f"{n:04d}"

    @classmethod
    def generate_key(cls, duration_days: int) -> str:
        if duration_days not in ALLOWED_DURATIONS:
            raise ValueError("Допустимые сроки: 7, 14 или 30 дней")

        b5 = cls._block_multiple_of(5)
        b8 = cls._block_multiple_of(8)
        b3 = cls._block_multiple_of(3)
        b9 = cls._block_multiple_of(9)

        base = f"{cls.PREFIX}-{duration_days}-{b5}-{b8}-{b3}-{b9}"
        checksum = cls._checksum(base)
        return f"{base}-{checksum}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация лицензионных ключей FishingBot")
    parser.add_argument("--days", type=int, required=True, choices=ALLOWED_DURATIONS, help="Срок ключа: 7, 14, 30")
    parser.add_argument("--count", type=int, default=1, help="Количество ключей")
    args = parser.parse_args()

    if args.count < 1:
        raise ValueError("count должен быть >= 1")

    for _ in range(args.count):
        print(LicenseKeyGenerator.generate_key(args.days))


if __name__ == "__main__":
    main()

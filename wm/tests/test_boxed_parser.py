"""\boxed{} ayrıştırıcı testi (CPU): iç içe parantez, birim, kesir, kesilmiş üretim.

  python -m wm.tests.test_boxed_parser
"""
from eval.aime_tr import last_boxed_num, parse_boxed_content

CASES = [
    (r"... \boxed{42}", (42.0, True)),
    (r"önce \boxed{1} sonra \boxed{216}", (216.0, True)),          # SON kutu
    (r"\boxed{\frac{1}{2}}", (0.5, True)),
    (r"\boxed{\dfrac{3}{4}}", (0.75, True)),
    (r"\boxed{12\ \text{cm}}", (12.0, True)),
    (r"\boxed{9{,}720}", (None, True)),                              # sayı değil → None
    (r"\boxed{9.720}", (9.72, True)),
    (r"\boxed{1,234}", (1234.0, True)),
    (r"\boxed{3,14}", (3.14, True)),
    (r"\boxed{45^\circ}", (45.0, True)),
    (r"\boxed{25\%}", (25.0, True)),
    (r"\boxed{x+1}", (None, True)),
    (r"cevap 42", (None, False)),
    (r"\boxed{12", (None, False)),                                   # kesilmiş
]
for text, want in CASES:
    got = last_boxed_num(text)
    assert got == want, (text, got, want)
assert parse_boxed_content("-7") == -7.0
print(f"ok  {len(CASES)} ayrıştırma durumu")
print("BOXED TESTİ GEÇTİ")

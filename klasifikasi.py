"""
klasifikasi.py - Algoritma Multinomial Naive Bayes + aturan pendukung kelas Netral.
Aturan memakai kamus kata (leksikon) sederhana; TIDAK dilatih dari data, jadi tidak mengubah model .pkl.
"""
POSITIF = set("""bagus baik bagusss baguss mantap mantab mantul mantull mantaabb puas suka senang cepat mudah gampang lancar
keren kece recommended rekomen recommend murah ramah membantu memudahkan original ori lengkap praktis terbaik favorit
memuaskan nyaman simple simpel sempurna top oke ok good great nice easy best love loved excellent amazing awesome helpful
fast smooth friendly perfect thanks thank terimakasih trusted worth hemat untung diskon promo menarik enak lega aman rapi
rapih responsif sip joss""".split())
NEGATIF = set("""jelek buruk lemot lambat lama error eror bug crash gagal kecewa parah ribet susah mahal rusak penipuan penipu tipu
sampah bohong batal dibatalkan hilang worst bad slow terrible awful useless disappointed scam fraud lag hang ngelag macet
sulit menyebalkan mengecewakan kapok payah zonk cacat telat terlambat lelet nyebelin jengkel marah komplain keluhan rugi
masalah bermasalah susahnya sakit kurang annoying problem issue fail failed cancel cancelled canceled spam sepihak ditolak
palsu menipu""".split())
RAGU = set("""biasa lumayan standar cukup sedang average kadang sesekali sometimes so-so okelah""".split())
KONTRAS = set("tapi tetapi namun cuma cuman walaupun meskipun but however though although sayangnya".split())
NEGATOR = set("tidak gak ga nggak ngga enggak bukan belum tak never not no kurang".split())
MAKS_KATA_TANPA_OPINI = 6


def hitung_fitur(teks_bersih):
    t = teks_bersih.split()
    pos = neg = 0
    for i, w in enumerate(t):
        dinegasi = i > 0 and t[i - 1] in NEGATOR
        if w in POSITIF:
            neg, pos = (neg + 1, pos) if dinegasi else (neg, pos + 1)
        elif w in NEGATIF:
            neg, pos = (neg, pos + 1) if dinegasi else (neg + 1, pos)
    ragu = sum(w in RAGU for w in t)
    kontras = sum(w in KONTRAS for w in t)
    ada_negasi = any(w in NEGATOR for w in t)
    return pos, neg, ragu, kontras, len(t), ada_negasi


def putusan(label_nb, teks_bersih):
    """Mengembalikan (label_akhir, dasar_keputusan)."""
    pos, neg, ragu, kontras, n, ada_negasi = hitung_fitur(teks_bersih)
    if ragu > 0 and pos + neg <= 2:
        return "Netral", "Aturan: kata bernada sedang (biasa/lumayan/cukup/kadang)"
    if pos > 0 and neg > 0 and kontras > 0:
        return "Netral", "Aturan: opini campuran positif dan negatif"
    if pos == 0 and neg == 0 and ragu == 0 and n <= MAKS_KATA_TANPA_OPINI and not ada_negasi:
        return "Netral", "Aturan: tidak ada kata opini (kalimat informatif)"
    if pos > 0 and neg == 0 and ragu == 0 and label_nb != "Positif":
        return "Positif", "Aturan: hanya kata bernada positif"
    if neg > 0 and pos == 0 and ragu == 0 and label_nb != "Negatif":
        return "Negatif", "Aturan: hanya kata bernada negatif"
    return label_nb, "Algoritma Naive Bayes"

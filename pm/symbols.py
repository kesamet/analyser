"""
Symbols
"""
others_dict = {
    "IWDA.L": "IWDA",
    "EIMI.L": "EIMI",
    "USDSGD=X": "USDSGD",
    "SGDIDR=X": "SGDIDR",
    "^VIX": "VIX",
    # "^GSPC": "SP500",
    # "^STI": "STI",
    "ES3.SI": "STI ETF",
    "ACWI": "iShares MSCI ACWI ETF",
    "URTH": "iShares MSCI World ETF",
    "IEUR": "iShares Core MSCI Europe ETF",
    "EEM": "iShares MSCI Emerging Markets ETF",
    "AAXJ": "iShares MSCI All Country Asia ex Jpn ETF",
    "VWRD.L": "Vanguard FTSE ALL-WORLD",
    "SPY": "SPDR S&P 500 ETF",
    "G3B.SI": "Nikko STI ETF",
    "CLR.SI": "Lion-Phillip S-REIT ETF",
    # "A35.SI": "ABF SG Bond ETF",
    "GOTO.JK": "GoTo",
    "GRAB": "Grab",
    "D05.SI": "DBS Group Holdings Ltd",
    "O39.SI": "Oversea-Chinese Banking Corp Ltd",
    "U11.SI": "United Overseas Bank Ltd",
}

reits_dict = {
    "O5RU.SI": "AIMS APAC REIT",
    "XZL.SI": "ARA US Hospitality Trust",
    "BMGU.SI": "BHG Retail REIT",
    "A17U.SI": "Capitaland Ascendas REIT",
    "HMN.SI": "Capitaland Ascott Trust",
    "AU8U.SI": "Capitaland China Trust",
    "C38U.SI": "Capitaland Integrated Commercial Trust",
    "J85.SI": "CDL Hospitality Trust",
    # "CNNU.SI": "Cromwell European REIT",
    # "DHLU.SI": "Daiwa House Logistics Trust",
    # "DCRU.SI": "Digital Core REIT",
    "BWCU.SI": "EC World REIT",
    "MXNU.SI": "Elite Commercial REIT",
    "J91U.SI": "ESR-Logos REIT",
    "Q5T.SI": "Far East Hospitality Trust",
    "AW9U.SI": "First REIT",
    "J69U.SI": "Frasers Centrepoint Trust",
    "ACV.SI": "Frasers Hospitality Trust",
    "BUOU.SI": "Frasers L&C Trust",
    "UD1U.SI": "IREIT Global",
    "AJBU.SI": "Keppel DC REIT",
    "CMOU.SI": "Keppel Pacific Oak US RIET",
    "K71U.SI": "Keppel REIT",
    "JYEU.SI": "Lendlease Global Comm REIT",
    "D5IU.SI": "Lippo Malls Indonesia Retail Trust",
    "BTOU.SI": "Manulife US REIT",
    "ME8U.SI": "Mapletree Industrial Trust",
    "M44U.SI": "Mapletree Logistics Trust",
    "N2IU.SI": "Mapletree Pan Asia Commercial Trust",
    "TS0U.SI": "OUE Commercial REIT",
    "SK6U.SI": "Paragon REIT",
    "C2PU.SI": "Parkway Life REIT",
    "OXMU.SI": "Prime US REIT",
    "M1GU.SI": "Sabana REIT",
    "CRPU.SI": "Sasseur REIT",
    "P40U.SI": "Starhill Global REIT",
    "T82U.SI": "Suntec REIT",
    # "ODBU.SI": "United Hampshire US REIT",
    "CJLU.SI": "NetLink NBN Trust",
}

SYMBOLS = list(others_dict.keys()) + list(reits_dict.keys())

# Selected
dct = {
    "IWDA.L": "IWDA",
    "EIMI.L": "EIMI",
    "GOTO.JK": "GoTo",
    "ES3.SI": "STI ETF",
}
dct.update(reits_dict)
EQ_DICT = {v: k for k, v in dct.items()}

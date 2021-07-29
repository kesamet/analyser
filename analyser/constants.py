"""
Constants and parameters
"""
str2days = {
    "1M": 30,
    "2M": 60,
    "3M": 91,
    "6M": 182,
    "9M": 273,
    "1Y": 365,
    "2Y": 730,
}

PHRASES_SEARCH = {
        "Aggregate leverage, Gearing": "or",
        "Cost of debt": "or",
        "Interest cover": "or",
        "Average term to maturity": "or",
        "WALE, Weighted average lease expiry": "or",
        "Net property income": "or",
        "Distribution per unit, DPU": "or",
        "Total debt": "or",
        "Total assets": "or",
        "Total liabilities": "or",
        "Investment properties": "or",
        "Units in issue": "or",
        "Net asset value, NAV": "or",
        "Unit price performance, Closing, Highest, Lowest": "and",
        "Financial position, Total assets, Total liabilities, Investment properties": "and",
    }

KEYWORDS_EXTRACT_SLIDES = {
    "Net property income": {
        "keywords": ["Net property income"],
    },
    "Distribution per unit": {
        "keywords": ["Distribution per unit", "DPU"],
    },
    "Total assets": {
        "keywords": ["Total assets"],
    },
    "Total liabilities": {
        "keywords": ["Total liabilities"],
    },
    "Total debt": {
        "keywords": ["Total debt"],
    },
    "Units in issue": {
        "keywords": ["Units in issue"],
    },
    "Net asset value": {
        "keywords": ["Net asset value", "NAV"],
    },
    "Aggregate leverage": {
        "keywords": ["Aggregate leverage", "Gearing"],
    },
    "Cost of debt": {
        "keywords": ["Cost of debt"],
    },
    "Interest cover": {
        "keywords": ["Interest cover"],
    },
    "Average term to maturity": {
        "keywords": ["Average term to maturity", "Average debt maturity"],
    },
    "WALE": {
        "keywords": ["WALE", "Weighted average lease expiry"],
    },
}

KEYWORDS_EXTRACT_REPORT = {
    "Net property income": {
        "keywords": ["Net property income"],
        "aux_kws": [
            "Review of performance",
            "1(a)(i) Statement of total return",
            "1(a)(i) Statements of total return",
        ],
    },
    "Distribution per unit": {
        "keywords": ["Distribution per unit", "DPU"],
        "aux_kws": [
            "Review of performance",
            "6 Earnings per Unit",
            "6. Earnings per Unit",
        ],
    },
    "Investment properties": {
        "keywords": ["Investment properties"],
        "aux_kws": [
            "Statement of Financial Position",
            "Statements of Financial Position",
        ],
    },
    "Total assets": {
        "keywords": ["Total assets"],
        "aux_kws": [
            "Statement of Financial Position",
            "Statements of Financial Position",
        ],
    },
    "Total liabilities": {
        "keywords": ["Total liabilities"],
        "aux_kws": [
            "Statement of Financial Position",
            "Statements of Financial Position",
        ],
    },
    "Perpetual securities": {
        "keywords": ["Perpetual securities"],
        "aux_kws": [
            "Statement of Financial Position",
            "Statements of Financial Position",
        ],
    },
    "Units": {
        "keywords": ["Units issued", "Issued Units", "Total issued and issuable Units"],
        "aux_kws": [
            "1(d)(ii) Details of",
        ],
    },
    "Net asset value": {
        "keywords": ["NAV", "Net asset value"],
        "aux_kws": [
            "7 Net Asset Value",
            "7. Net Asset Value",
        ],
    },
}

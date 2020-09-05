search_words = [
    "Net property income",
    "Distribution per unit", "DPU",
    "Investment properties",
    "Total assets",
    "Total liabilities",
    "Perpetual securities",
    "Total debts",
    "Units", "Units in issue",
    "Net asset value", "NAV",
    "Aggregate leverage", "Gearing",
    "Cost of debt",
    "Interest cover",
    "Average term to maturity",
    "WALE", "Weighted average",
    "Unit price performance",
    "Total return",
    "Distribution",
    "Financial position",
]

search_words2 = [
    "Total return,Net property income",
    "Distribution statement,Distribution per unit",
    "Financial position,Total assets,Total liabilities,Investment properties",
    "Aggregate leverage,Cost of debt,Interest cover,Average term to maturity",
    "Unit price performance,Closing,Highest,Lowest",
]


dct = {
    "Ascendas": {
        "Net property income": {
            "title": "1(a)(i) Statement of Total Return and Distribution Statement",
            "ending": "Page",
        },
        "Distribution per unit": {
            "title": "6. Earnings per Unit (“EPU”) and Distribution per Unit (“DPU”) for the financial period",
            "ending": "Page",
        },
        "Investment properties": {
            "title": "1(b)(i) Statements of Financial Position",
            "ending": "Page",
        },
        "Total assets": {
            "title": "1(b)(i) Statements of Financial Position",
            "ending": "Page",
        },
        "Total liabilities": {
            "title": "1(b)(i) Statements of Financial Position",
            "ending": "Page",
        },
        "Perpetual securities holders": {
            "title": "1(b)(i) Statements of Financial Position",
            "ending": "Page",
        },
        "Units issued and issuable at end of the financial period": {
            "title": "1(d)(ii) Details of any changes in the Units",
            "ending": "1(d)(iii)",
        },
        "Net asset value per Unit": {
            "title": "7. Net asset value per Unit based on Units issued at the end of the period",
            "ending": "Page",
        },
    },
}

# CRCT
# "Statements of total return", "Net property income"
# "Distribution statements", "Distribution per unit"
# "Statements of financial position", "Total assets"
# "Units in issue", "Number of units"
#
# "Unit price performance", "closing", "highest", "lowest"
# "Key financial indicators", "Aggregate leverage", "Interest coverage", "Average term to maturity", "Cost of debt"

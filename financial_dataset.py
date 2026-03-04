"""
Financial statements dataset for FinancialModernBert.

This module handles:
1. Converting JSON financial statements to Markdown tables with <number></number> tags
2. Creating a PyTorch Dataset that masks ~10% of numbers for training
"""
import json
import os
import random
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from torch.utils.data import Dataset
import torch
from functools import partial


# Define the financial statement fields in a meaningful order
# These are the key metrics from income statement, balance sheet, and cash flow
FINANCIAL_FIELDS = [
    # Income Statement
    "revenue",
    "costOfRevenue", 
    "grossProfit",
    "grossProfitRatio",
    "researchAndDevelopmentExpenses",
    "generalAndAdministrativeExpenses",
    "sellingAndMarketingExpenses",
    "sellingGeneralAndAdministrativeExpenses",
    "otherExpenses",
    "operatingExpenses",
    "costAndExpenses",
    "interestIncome",
    "interestExpense",
    "depreciationAndAmortization",
    "ebitda",
    "ebitdaratio",
    "operatingIncome",
    "operatingIncomeRatio",
    "totalOtherIncomeExpensesNet",
    "incomeBeforeTax",
    "incomeBeforeTaxRatio",
    "incomeTaxExpense",
    "netIncome",
    "netIncomeRatio",
    "eps",
    "epsdiluted",
    "weightedAverageShsOut",
    "weightedAverageShsOutDil",
    # Balance Sheet - Assets
    "cashAndCashEquivalents",
    "shortTermInvestments",
    "cashAndShortTermInvestments",
    "netReceivables",
    "inventory",
    "otherCurrentAssets",
    "totalCurrentAssets",
    "propertyPlantEquipmentNet",
    "goodwill",
    "intangibleAssets",
    "goodwillAndIntangibleAssets",
    "longTermInvestments",
    "taxAssets",
    "otherNonCurrentAssets",
    "totalNonCurrentAssets",
    "otherAssets",
    "totalAssets",
    # Balance Sheet - Liabilities
    "accountPayables",
    "shortTermDebt",
    "taxPayables",
    "deferredRevenue",
    "otherCurrentLiabilities",
    "totalCurrentLiabilities",
    "longTermDebt",
    "deferredRevenueNonCurrent",
    "deferredTaxLiabilitiesNonCurrent",
    "otherNonCurrentLiabilities",
    "totalNonCurrentLiabilities",
    "otherLiabilities",
    "capitalLeaseObligations",
    "totalLiabilities",
    # Balance Sheet - Equity
    "preferredStock",
    "commonStock",
    "retainedEarnings",
    "accumulatedOtherComprehensiveIncomeLoss",
    "othertotalStockholdersEquity",
    "totalStockholdersEquity",
    "totalEquity",
    "totalLiabilitiesAndStockholdersEquity",
    "minorityInterest",
    "totalLiabilitiesAndTotalEquity",
    "totalInvestments",
    "totalDebt",
    "netDebt",
    # Cash Flow Statement
    "deferredIncomeTax",
    "stockBasedCompensation",
    "changeInWorkingCapital",
    "accountsReceivables",
    "accountsPayables",
    "otherWorkingCapital",
    "otherNonCashItems",
    "netCashProvidedByOperatingActivities",
    "investmentsInPropertyPlantAndEquipment",
    "acquisitionsNet",
    "purchasesOfInvestments",
    "salesMaturitiesOfInvestments",
    "otherInvestingActivites",
    "netCashUsedForInvestingActivites",
    "debtRepayment",
    "commonStockIssued",
    "commonStockRepurchased",
    "dividendsPaid",
    "otherFinancingActivites",
    "netCashUsedProvidedByFinancingActivities",
    "effectOfForexChangesOnCash",
    "netChangeInCash",
    "cashAtEndOfPeriod",
    "cashAtBeginningOfPeriod",
    "operatingCashFlow",
    "capitalExpenditure",
    "freeCashFlow",
]

# Human-readable names for the fields
FIELD_NAMES = {
    "revenue": "Revenue",
    "costOfRevenue": "Cost of Revenue",
    "grossProfit": "Gross Profit",
    "grossProfitRatio": "Gross Profit Ratio",
    "researchAndDevelopmentExpenses": "R&D Expenses",
    "generalAndAdministrativeExpenses": "G&A Expenses",
    "sellingAndMarketingExpenses": "S&M Expenses",
    "sellingGeneralAndAdministrativeExpenses": "SG&A Expenses",
    "otherExpenses": "Other Expenses",
    "operatingExpenses": "Operating Expenses",
    "costAndExpenses": "Cost and Expenses",
    "interestIncome": "Interest Income",
    "interestExpense": "Interest Expense",
    "depreciationAndAmortization": "D&A",
    "ebitda": "EBITDA",
    "ebitdaratio": "EBITDA Ratio",
    "operatingIncome": "Operating Income",
    "operatingIncomeRatio": "Operating Income Ratio",
    "totalOtherIncomeExpensesNet": "Other Income/Expenses Net",
    "incomeBeforeTax": "Income Before Tax",
    "incomeBeforeTaxRatio": "Income Before Tax Ratio",
    "incomeTaxExpense": "Income Tax Expense",
    "netIncome": "Net Income",
    "netIncomeRatio": "Net Income Ratio",
    "eps": "EPS",
    "epsdiluted": "EPS Diluted",
    "weightedAverageShsOut": "Weighted Avg Shares Out",
    "weightedAverageShsOutDil": "Weighted Avg Shares Out Diluted",
    "cashAndCashEquivalents": "Cash and Cash Equivalents",
    "shortTermInvestments": "Short-Term Investments",
    "cashAndShortTermInvestments": "Cash and Short-Term Investments",
    "netReceivables": "Net Receivables",
    "inventory": "Inventory",
    "otherCurrentAssets": "Other Current Assets",
    "totalCurrentAssets": "Total Current Assets",
    "propertyPlantEquipmentNet": "Property Plant Equipment Net",
    "goodwill": "Goodwill",
    "intangibleAssets": "Intangible Assets",
    "goodwillAndIntangibleAssets": "Goodwill and Intangible Assets",
    "longTermInvestments": "Long-Term Investments",
    "taxAssets": "Tax Assets",
    "otherNonCurrentAssets": "Other Non-Current Assets",
    "totalNonCurrentAssets": "Total Non-Current Assets",
    "otherAssets": "Other Assets",
    "totalAssets": "Total Assets",
    "accountPayables": "Accounts Payables",
    "shortTermDebt": "Short-Term Debt",
    "taxPayables": "Tax Payables",
    "deferredRevenue": "Deferred Revenue",
    "otherCurrentLiabilities": "Other Current Liabilities",
    "totalCurrentLiabilities": "Total Current Liabilities",
    "longTermDebt": "Long-Term Debt",
    "deferredRevenueNonCurrent": "Deferred Revenue Non-Current",
    "deferredTaxLiabilitiesNonCurrent": "Deferred Tax Liabilities Non-Current",
    "otherNonCurrentLiabilities": "Other Non-Current Liabilities",
    "totalNonCurrentLiabilities": "Total Non-Current Liabilities",
    "otherLiabilities": "Other Liabilities",
    "capitalLeaseObligations": "Capital Lease Obligations",
    "totalLiabilities": "Total Liabilities",
    "preferredStock": "Preferred Stock",
    "commonStock": "Common Stock",
    "retainedEarnings": "Retained Earnings",
    "accumulatedOtherComprehensiveIncomeLoss": "Accumulated OCI",
    "othertotalStockholdersEquity": "Other Stockholders Equity",
    "totalStockholdersEquity": "Total Stockholders Equity",
    "totalEquity": "Total Equity",
    "totalLiabilitiesAndStockholdersEquity": "Total Liabilities and Stockholders Equity",
    "minorityInterest": "Minority Interest",
    "totalLiabilitiesAndTotalEquity": "Total Liabilities and Total Equity",
    "totalInvestments": "Total Investments",
    "totalDebt": "Total Debt",
    "netDebt": "Net Debt",
    "deferredIncomeTax": "Deferred Income Tax",
    "stockBasedCompensation": "Stock-Based Compensation",
    "changeInWorkingCapital": "Change in Working Capital",
    "accountsReceivables": "Accounts Receivables",
    "accountsPayables": "Accounts Payables (CF)",
    "otherWorkingCapital": "Other Working Capital",
    "otherNonCashItems": "Other Non-Cash Items",
    "netCashProvidedByOperatingActivities": "Net Cash from Operating",
    "investmentsInPropertyPlantAndEquipment": "Investments in PP&E",
    "acquisitionsNet": "Acquisitions Net",
    "purchasesOfInvestments": "Purchases of Investments",
    "salesMaturitiesOfInvestments": "Sales/Maturities of Investments",
    "otherInvestingActivites": "Other Investing Activities",
    "netCashUsedForInvestingActivites": "Net Cash from Investing",
    "debtRepayment": "Debt Repayment",
    "commonStockIssued": "Common Stock Issued",
    "commonStockRepurchased": "Common Stock Repurchased",
    "dividendsPaid": "Dividends Paid",
    "otherFinancingActivites": "Other Financing Activities",
    "netCashUsedProvidedByFinancingActivities": "Net Cash from Financing",
    "effectOfForexChangesOnCash": "Effect of Forex on Cash",
    "netChangeInCash": "Net Change in Cash",
    "cashAtEndOfPeriod": "Cash at End of Period",
    "cashAtBeginningOfPeriod": "Cash at Beginning of Period",
    "operatingCashFlow": "Operating Cash Flow",
    "capitalExpenditure": "Capital Expenditure",
    "freeCashFlow": "Free Cash Flow",
}


def format_number(value: Any) -> str:
    """Format a number with <number></number> tags."""
    if value is None:
        return "<number>0</number>"
    try:
        num = float(value)
        # Format with reasonable precision
        if abs(num) >= 1:
            formatted = f"{num:.2f}"
        else:
            formatted = f"{num:.6f}"
        return f"<number>{formatted}</number>"
    except (ValueError, TypeError):
        return "<number>0</number>"


def financial_statement_to_markdown(
    reports: List[Dict[str, Any]], 
    max_years: int = 3,
    shuffle_rows: bool = False
) -> Optional[str]:
    """
    Convert a list of financial reports (sorted by date, most recent first) 
    to a Markdown table with <number></number> tags.
    
    Args:
        reports: List of report dicts for the same company, sorted by date descending
        max_years: Maximum number of years to include (default 3)
        shuffle_rows: Whether to shuffle the rows (data augmentation)
    
    Returns:
        Markdown table string, or None if insufficient data
    """
    if not reports or len(reports) < max_years:
        return None
    
    # Take only the most recent years
    reports = reports[:max_years]
    
    # Get calendar years for column headers
    years = []
    for r in reports:
        year = r.get("calendarYear", r.get("date", "")[:4])
        years.append(str(year))
    
    # Build header row with actual years
    if len(reports) == 3:
        header = f"| Item | {years[0]} | {years[1]} | {years[2]} |"
        separator = "|---|---|---|---|"
    elif len(reports) == 2:
        header = f"| Item | {years[0]} | {years[1]} |"
        separator = "|---|---|---|"
    else:
        header = f"| Item | {years[0]} |"
        separator = "|---|---|"
    
    # Build data rows
    rows = []
    fields_to_use = FINANCIAL_FIELDS.copy()
    
    if shuffle_rows:
        random.shuffle(fields_to_use)
    
    for field in fields_to_use:
        field_name = FIELD_NAMES.get(field, field)
        values = []
        
        has_nonzero = False
        for r in reports:
            val = r.get(field, 0)
            if val is not None and val != 0:
                has_nonzero = True
            values.append(format_number(val))
        
        # Skip rows where all values are zero (reduces noise)
        if not has_nonzero:
            continue
        
        row = f"| {field_name} | " + " | ".join(values) + " |"
        rows.append(row)
    
    if len(rows) < 5:  # Skip if too few meaningful rows
        return None
    
    # Construct final markdown
    table = header + "\n" + separator + "\n" + "\n".join(rows)
    return table


def load_financial_data(data_dir: str, file_indices: Optional[List[int]] = None) -> Dict[str, List[Dict]]:
    """
    Load financial data from JSON files.
    
    Args:
        data_dir: Directory containing full_reports_*.json files
        file_indices: List of file indices to load (default: 0-25)
    
    Returns:
        Dictionary mapping symbol -> list of reports
    """
    if file_indices is None:
        file_indices = list(range(26))  # 0 to 25
    
    all_data = {}
    
    for idx in file_indices:
        file_path = Path(data_dir) / f"full_reports_{idx}.json"
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping...")
            continue
        
        print(f"Loading {file_path}...")
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        for symbol, reports in data.items():
            if symbol not in all_data:
                all_data[symbol] = []
            all_data[symbol].extend(reports)
    
    # Sort reports by date (most recent first) for each symbol
    for symbol in all_data:
        all_data[symbol].sort(key=lambda x: x.get("date", ""), reverse=True)
    
    return all_data


class FinancialStatementDataset(Dataset):
    """
    Dataset for training FinancialModernBert on financial statements.
    
    Each sample is a markdown table of a financial statement where ~10% of numbers
    are masked for the model to predict.
    """
    
    def __init__(
        self,
        data_dir: str,
        tokenizer,
        file_indices: Optional[List[int]] = None,
        mask_ratio: float = 0.10,
        max_years: int = 3,
        shuffle_rows: bool = False,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.mask_ratio = mask_ratio
        self.max_years = max_years
        self.shuffle_rows = shuffle_rows
        self.max_length = max_length
        
        # Load and convert data
        print("Loading financial data...")
        raw_data = load_financial_data(data_dir, file_indices)
        
        # Convert to markdown tables - create multiple samples per company
        # by generating all possible consecutive windows of max_years
        print("Converting to markdown tables...")
        self.samples = []
        for symbol, reports in raw_data.items():
            # Generate all possible consecutive windows
            # For example, if we have 5 years and max_years=3:
            # Window 1: years 0,1,2 (most recent)
            # Window 2: years 1,2,3
            # Window 3: years 2,3,4
            num_windows = len(reports) - max_years + 1
            
            if num_windows < 1:
                # Not enough data for even one window
                continue
            
            for window_idx in range(num_windows):
                window_reports = reports[window_idx:window_idx + max_years]
                md_table = financial_statement_to_markdown(
                    window_reports, 
                    max_years=max_years,
                    shuffle_rows=shuffle_rows
                )
                if md_table:
                    self.samples.append({
                        "symbol": symbol,
                        "markdown": md_table,
                        "reports": window_reports
                    })
        
        print(f"Created {len(self.samples)} samples from {len(raw_data)} companies")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "markdown": sample["markdown"],
            "symbol": sample["symbol"]
        }


def financial_collate_fn(batch: List[Dict], tokenizer, mask_ratio: float = 0.10, max_length: int = 2048):
    """
    Collate function for financial statement training.
    
    Masks ~mask_ratio of number tokens and prepares labels.
    """
    import re
    
    texts = [item['markdown'] for item in batch]
    
    # First, tokenize to find all number positions
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    input_ids = encodings['input_ids'].clone()
    is_number_mask = encodings['is_number_mask'].clone()
    number_values = encodings['number_values'].clone()
    attention_mask = encodings['attention_mask']
    
    batch_size, seq_len = input_ids.shape
    
    # Initialize labels (all -100 means ignore)
    labels_text = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    labels_magnitude = torch.full((batch_size, seq_len), -100.0, dtype=torch.float)
    labels_sign = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    
    mask_token_id = tokenizer.mask_token_id
    
    # For each sample, mask some of the numbers
    for i in range(batch_size):
        # Find all number positions in this sample
        number_positions = (is_number_mask[i] == 1).nonzero(as_tuple=True)[0]
        
        if len(number_positions) == 0:
            continue
        
        # Determine how many to mask
        num_to_mask = max(1, int(len(number_positions) * mask_ratio))
        
        # Randomly select positions to mask
        mask_indices = random.sample(list(number_positions.tolist()), num_to_mask)
        
        for pos in mask_indices:
            # Store the ground truth
            sign_val = int(number_values[i, pos, 0].item())
            mag_val = number_values[i, pos, 1].item()
            
            labels_sign[i, pos] = sign_val
            labels_magnitude[i, pos] = mag_val
            
            # Replace with mask token (the model will see this as a number to predict)
            # We keep is_number_mask[i, pos] = 1 so the model knows it's a number position
            # But we zero out the actual number value so it can't "cheat"
            number_values[i, pos, 0] = 0.0
            number_values[i, pos, 1] = 0.0
            
            # Also replace input_id with mask token
            input_ids[i, pos] = mask_token_id
    
    return {
        "input_ids": input_ids,
        "is_number_mask": is_number_mask,
        "number_values": number_values,
        "attention_mask": attention_mask,
        "labels_text": labels_text,
        "labels_magnitude": labels_magnitude,
        "labels_sign": labels_sign
    }


def create_train_val_split(
    data_dir: str,
    tokenizer,
    train_ratio: float = 0.9,
    file_indices: Optional[List[int]] = None,
    mask_ratio: float = 0.10,
    max_years: int = 3,
    max_length: int = 2048,
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dict[str, Any]]:
    """
    Create train/validation split of the financial dataset.
    
    Args:
        data_dir: Directory containing the financial data
        tokenizer: FinancialBertTokenizer instance
        train_ratio: Fraction of data for training (default 0.9)
        file_indices: Which file indices to load
        mask_ratio: Fraction of numbers to mask
        max_years: Max years to include per statement
        max_length: Max sequence length
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_dataset, val_dataset, split_info)
        split_info contains: train_symbols, val_symbols, file_indices, seed
    """
    random.seed(seed)
    
    # Create full dataset
    full_dataset = FinancialStatementDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        file_indices=file_indices,
        mask_ratio=mask_ratio,
        max_years=max_years,
        max_length=max_length
    )
    
    symbol_to_indices = {}
    for idx, sample in enumerate(full_dataset.samples):
        symbol = sample['symbol']
        if symbol not in symbol_to_indices:
            symbol_to_indices[symbol] = []
        symbol_to_indices[symbol].append(idx)
    
    all_symbols = list(symbol_to_indices.keys())
    random.shuffle(all_symbols)
    
    split_idx = int(len(all_symbols) * train_ratio)
    train_symbols = all_symbols[:split_idx]
    val_symbols = all_symbols[split_idx:]
    
    train_symbols_set = set(train_symbols)
    val_symbols_set = set(val_symbols)
    
    train_indices = []
    val_indices = []
    for symbol, indices in symbol_to_indices.items():
        if symbol in train_symbols_set:
            train_indices.extend(indices)
        else:
            val_indices.extend(indices)
    
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    split_info = {
        'train_symbols': train_symbols,
        'val_symbols': val_symbols,
        'file_indices': file_indices if file_indices else list(range(26)),
        'seed': seed,
        'train_ratio': train_ratio,
    }
    
    print(f"Train: {len(train_dataset)} samples from {len(train_symbols)} companies")
    print(f"Val:   {len(val_dataset)} samples from {len(val_symbols)} companies")
    
    return train_dataset, val_dataset, split_info


def save_split_info(split_info: Dict[str, Any], filepath: str):
    """Save split info to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"Split info saved to {filepath}")


def load_split_info(filepath: str) -> Dict[str, Any]:
    """Load split info from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_validation_dataset(
    data_dir: str,
    tokenizer,
    split_info: Dict[str, Any],
    mask_ratio: float = 0.10,
    max_years: int = 3,
    max_length: int = 2048,
) -> Dataset:
    """
    Load ONLY the validation dataset using a previously saved split.
    
    This ensures validation uses the exact same companies as during training,
    preventing any data leakage.
    
    Args:
        data_dir: Directory containing the financial data
        tokenizer: FinancialBertTokenizer instance
        split_info: Dictionary with 'val_symbols' and 'file_indices' from training
        mask_ratio: Fraction of numbers to mask
        max_years: Max years to include per statement
        max_length: Max sequence length
    
    Returns:
        Validation dataset containing only companies from split_info['val_symbols']
    """
    file_indices = split_info.get('file_indices', list(range(26)))
    val_symbols_set = set(split_info['val_symbols'])
    
    # Create full dataset with the same file indices used during training
    full_dataset = FinancialStatementDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        file_indices=file_indices,
        mask_ratio=mask_ratio,
        max_years=max_years,
        max_length=max_length
    )
    
    # Filter to only validation symbols
    val_indices = []
    for idx, sample in enumerate(full_dataset.samples):
        if sample['symbol'] in val_symbols_set:
            val_indices.append(idx)
    
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    print(f"Loaded validation dataset: {len(val_dataset)} samples from {len(val_symbols_set)} companies")
    
    return val_dataset


# if __name__ == "__main__":
#     # Quick test
#     from financial_bert import FinancialBertTokenizer
    
#     tokenizer = FinancialBertTokenizer()
    
#     # Test with just one file
#     data_dir = "financial_statements_data"
    
#     train_ds, val_ds, split_info = create_train_val_split(
#         data_dir=data_dir,
#         tokenizer=tokenizer,
#         file_indices=[25],  # Just the lightest file for testing
#         mask_ratio=0.10
#     )

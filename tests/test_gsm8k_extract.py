import pytest
from utils.eval_metrics_train import extract_final_number


def test_extract_final_number_commas():
    assert extract_final_number("1,200") == "1200"
    assert extract_final_number("$1,200.00") == "1200"
    assert extract_final_number("1,000,000") == "1000000"


def test_extract_final_number_negatives():
    assert extract_final_number("-3") == "-3"
    assert extract_final_number("Temperature dropped to -42 degrees.") == "-42"
    assert extract_final_number("#### -7") == "-7"


def test_extract_final_number_decimals():
    assert extract_final_number("2.5") == "2.5"
    assert extract_final_number("2.50") == "2.5"
    assert extract_final_number("3.0") == "3"
    assert extract_final_number("0.050") == "0.05"
    assert extract_final_number("-42.50") == "-42.5"


def test_extract_final_number_normalization_equivalence():
    assert extract_final_number("2.50") == extract_final_number("2.5")
    assert extract_final_number("3.0") == extract_final_number("3")
    assert extract_final_number("1,200") == extract_final_number("1200")
    assert extract_final_number("$1,200.00") == extract_final_number("1200")


def test_extract_final_number_canonical_gsm8k_hash():
    assert extract_final_number("#### 18") == "18"
    assert extract_final_number("#### 1,200") == "1200"
    assert extract_final_number("The total cost is #### 42.") == "42"
    assert extract_final_number("Intermediate step 10, then #### 25") == "25"


def test_extract_final_number_no_hash_takes_last_number():
    text = "Janet had 10 apples. She gave 3 to Bob and 2 to Charlie. She has 5 left."
    assert extract_final_number(text) == "5"


def test_extract_final_number_no_numbers():
    assert extract_final_number("There are no numbers in this sentence.") is None
    assert extract_final_number("") is None
    assert extract_final_number(None) is None


def test_evaluator_numeric_normalizer_config():
    from utils.eval_metrics_train import Evaluator
    cfg = {
        "evaluation": {
            "enabled": True,
            "strategy": "regex_extract",
            "answer_normalizer": "numeric",
            "answer_regex": "#### (\\d+)",
        }
    }
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.eval_cfg = cfg["evaluation"]
    evaluator.answer_normalizer = cfg["evaluation"].get("answer_normalizer")
    evaluator.strategy = cfg["evaluation"]["strategy"]
    assert evaluator.answer_normalizer == "numeric"
